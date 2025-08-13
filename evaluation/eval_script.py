#!/usr/bin/env python3
"""
InfiniBench Evaluation Script

This module provides automated evaluation capabilities for video-based question-answering tasks
using GPT-4o-mini for open-ended questions and accuracy metrics for multiple-choice questions.

Author: InfiniBench Team
License: See LICENSE.md
"""

import os
import sys
import json
import ast
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        client = None  # Will be checked later if needed
        logger.warning("OPENAI_API_KEY environment variable is not set")
    else:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
except Exception as e:
    client = None
    logger.error(f"Failed to initialize OpenAI client: {e}")


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Parsed JSON content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def gpt_score_wrapper(qa: Dict[str, Any], max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Evaluate a question-answer pair using GPT-4o-mini with comprehensive error handling and retry logic.
    
    Args:
        qa (Dict[str, Any]): Question-answer dictionary containing 'question', 'answer', and 'pred' keys
        max_retries (int): Maximum number of retry attempts for failed requests
        retry_delay (float): Base delay between retries in seconds (with exponential backoff)
        
    Returns:
        Dict[str, Any]: Updated QA dictionary with 'gpt_score' and 'gpt_justification' fields
    """
    if qa.get("gpt_score") is not None:
        logger.debug(f"Skipping already scored QA pair")
        return qa
    
    # Check if client is available
    if client is None:
        logger.error("OpenAI client not initialized. Please check your API key.")
        qa["gpt_score"] = None
        qa["gpt_justification"] = "OpenAI client not available"
        return qa
    
    # Validate required fields
    required_fields = ["question", "answer", "pred"]
    missing_fields = [field for field in required_fields if field not in qa]
    if missing_fields:
        logger.warning(f"Missing required fields: {missing_fields}")
        qa["gpt_score"] = None
        qa["gpt_justification"] = f"Missing fields: {missing_fields}"
        return qa
    
    question = qa["question"]
    answer = qa["answer"]
    pred = qa["pred"]
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent and fair evaluator AI that specializes in assessing the correctness and semantic alignment "
                            "between ground truth answers and predicted responses for question-answering tasks, including those based on video content.\n\n"
                            "Your role is to evaluate how well a predicted answer matches the correct (reference) answer based on the following detailed criteria:\n"
                            "------\n"
                            "## EVALUATION INSTRUCTIONS:\n"
                            "- Focus on **semantic similarity**, **factual correctness**, and **completeness**.\n"
                            "- Accept paraphrases, synonyms, or rephrasings **as valid**, as long as they preserve the original meaning.\n"
                            "- **Do not penalize** for stylistic differences or changes in tone, unless they impact factual accuracy.\n"
                            "- **Penalize** if:\n"
                            "  - The predicted answer omits **key factual elements** present in the correct answer.\n"
                            "  - The prediction includes **hallucinated content** or unfounded details.\n"
                            "  - The prediction **contradicts** the correct answer.\n"
                            "- Use human-like judgment: apply reasoning beyond surface text similarity.\n"
                            "- When uncertain, provide a **conservative but fair** score.\n"
                            "- Use a scoring scale from **0 (completely incorrect)** to **10 (perfect match)**.\n"
                            "## OUTPUT FORMAT:\n"
                            "Return a JSON object with **two fields**:\n"
                            '- "score": an integer from 0 to 10\n'
                            '- "justification": a concise explanation (1-3 sentences) of your reasoning\n\n'
                            "### Example Output:\n"
                            "{\n"
                            '  "score": 7,\n'
                            '  "justification": "The predicted answer captures the main idea, but it omits some key details about the setting described in the correct answer."\n'
                            "}\n"
                            "------\n"
                            "Be fair, consistent, and concise. Follow the format exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Please return your evaluation in the specified JSON format with both a score and a justification."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=500,
                timeout=30,  # 30 second timeout
            )
            
            response = completion.choices[0].message.content
            if isinstance(response, str):
                response_json = ast.literal_eval(response)
            
            # Validate response format
            if "score" not in response_json or "justification" not in response_json:
                raise ValueError(f"Invalid response format: {response_json}")
            
            # Validate score range
            score = response_json["score"]
            if not isinstance(score, int) or score < 0 or score > 10:
                raise ValueError(f"Invalid score: {score}. Must be integer between 0-10")
            
            qa["gpt_score"] = score
            qa["gpt_justification"] = response_json["justification"]
            logger.debug(f"Successfully scored QA pair with score: {score}")
            
            # Success - break out of retry loop
            break
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if this is a retryable error
            retryable_errors = [
                "connection error", "timeout", "rate limit", "server error", 
                "503", "502", "500", "429", "network", "connection"
            ]
            
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if attempt < max_retries and is_retryable:
                # Exponential backoff with jitter
                delay = retry_delay * (2 ** attempt) + (time.time() % 1)  # Add jitter
                logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                logger.info(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            else:
                # Final attempt failed or non-retryable error
                if attempt == max_retries:
                    logger.error(f"Failed to score QA pair after {max_retries + 1} attempts. Last error: {e}")
                else:
                    logger.error(f"Non-retryable error scoring QA pair: {e}")
                
                qa["gpt_score"] = None
                qa["gpt_justification"] = f"Evaluation failed: {str(e)}"
                break
    return qa


def eval_open_ended_skills(pred_dir: str, max_threads: int = 4, batch_size: int = 5, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate open-ended skills using GPT-4o-mini scoring with batch processing for safety.
    
    Args:
        pred_dir (str): Directory containing prediction files
        max_threads (int): Maximum number of threads for parallel processing
        batch_size (int): Number of items to process before saving (for safety)
        max_retries (int): Maximum number of retry attempts for failed requests
        retry_delay (float): Base delay between retries in seconds
        
    Returns:
        Dict[str, Dict[str, Any]]: Results dictionary with skill names as keys and metrics as values
    """
    skills_open_ended = [
        "summarization",
        "spoiler_questions", 
        "deep_context_understanding",
        "linking_multiple_events",
    ]
    
    logger.info("Starting evaluation of open-ended skills using GPT-4o-mini scoring...")
    
    
    if not os.path.exists(pred_dir):
        logger.error(f"Prediction directory does not exist: {pred_dir}")
        return {}
    
    skill_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    if not skill_files:
        logger.warning(f"No JSON files found in {pred_dir}")
        return {}
    
    # Dictionary to store results for summary
    open_ended_results = {}
    
    for skill_file_name in skill_files:
        skill_name = skill_file_name.split('.')[0]
        if skill_name not in skills_open_ended:
            logger.debug(f"Skipping {skill_name} (not an open-ended skill)")
            continue
            
        skill_path = os.path.join(pred_dir, skill_file_name)
        logger.info(f"{'-'*20}")
        logger.info(f"Processing skill: {skill_name}")
        logger.info(f"{'-'*20}")
        
        try:
            skill_data = read_json_file(skill_path)
        except Exception as e:
            logger.error(f"Failed to read {skill_path}: {e}")
            continue
        
        if not skill_data:
            logger.warning(f"No data found in {skill_path}")
            continue
        
        # Process in batches for safety
        updated_data = []
        total_batches = (len(skill_data) + batch_size - 1) // batch_size
        successful_evaluations = 0
        
        logger.info(f"Processing {len(skill_data)} items in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(skill_data))
            batch_data = skill_data[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            # Parallel scoring for this batch with reduced thread count if connection issues
            batch_results = []
            effective_threads = min(max_threads, len(batch_data), 2)  # Limit threads for stability
            
            with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                futures = [executor.submit(gpt_score_wrapper, qa.copy(), max_retries, retry_delay) for qa in batch_data]
                
                # Process completed futures as they finish
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per request
                        batch_results.append(result)
                        if result.get("gpt_score") is not None:
                            successful_evaluations += 1
                    except Exception as e:
                        logger.error(f"Future execution failed: {e}")
                        # Create a failed result entry
                        failed_result = batch_data[len(batch_results)].copy() if len(batch_results) < len(batch_data) else {}
                        failed_result["gpt_score"] = None
                        failed_result["gpt_justification"] = f"Request failed: {str(e)}"
                        batch_results.append(failed_result)
            
            # Add batch results to updated_data
            updated_data.extend(batch_results)
            
            # Save intermediate results for safety
            try:
                # Create backup before overwriting
                backup_path = f"{skill_path}.backup"
                if os.path.exists(skill_path):
                    with open(skill_path, 'r', encoding='utf-8') as original:
                        with open(backup_path, 'w', encoding='utf-8') as backup:
                            backup.write(original.read())
                
                # Combine processed data with remaining unprocessed data
                remaining_data = skill_data[end_idx:] if end_idx < len(skill_data) else []
                current_save_data = updated_data + remaining_data
                
                with open(skill_path, "w", encoding="utf-8") as f:
                    json.dump(current_save_data, f, indent=4, ensure_ascii=False)
                
                # Remove backup if save was successful
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    
                logger.info(f"Batch {batch_idx + 1} completed and saved. "
                           f"Success rate: {successful_evaluations}/{start_idx + len(batch_data)}")
                           
            except Exception as e:
                logger.error(f"Failed to save batch {batch_idx + 1}: {e}")
                # Restore from backup if available
                backup_path = f"{skill_path}.backup"
                if os.path.exists(backup_path):
                    os.rename(backup_path, skill_path)
                    logger.info("Restored from backup")
                continue
        
        # Calculate and display final statistics
        total_items = len(updated_data)
        missing_scores = sum(1 for qa in updated_data if qa.get("gpt_score") is None)
        valid_scores = [qa["gpt_score"] for qa in updated_data if qa.get("gpt_score") is not None]
        
        # Store results for summary
        if valid_scores:
            average_score = sum(valid_scores) / len(valid_scores)
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            open_ended_results[skill_name] = {
                'average_score': average_score,
                'min_score': min_score,
                'max_score': max_score,
                'total_items': total_items,
                'valid_scores': len(valid_scores),
                'missing_scores': missing_scores
            }
        else:
            open_ended_results[skill_name] = {
                'average_score': 0.0,
                'min_score': 0,
                'max_score': 0,
                'total_items': total_items,
                'valid_scores': 0,
                'missing_scores': missing_scores
            }
        
        if missing_scores > 0:
            logger.warning(f"{missing_scores}/{total_items} QA pairs had missing scores")
        
        if not valid_scores:
            logger.error(f"No valid scores obtained for {skill_name}")
    
    
    logger.info("Open-ended skills evaluation completed.")
    return open_ended_results


def mcq_accuracy(pred_data: List[Dict[str, Any]]) -> Tuple[float, int]:
    """
    Calculate accuracy for multiple-choice questions.
    
    Args:
        pred_data (List[Dict[str, Any]]): List of prediction dictionaries
        
    Returns:
        Tuple[float, int]: (accuracy, number_of_missing_predictions)
    """
    if not pred_data:
        return 0.0, 0
    
    correct_count = 0
    missing_qa = 0
    
    for qa in pred_data:
        if "pred" not in qa:
            missing_qa += 1
            continue
        if "answer_idx" not in qa:
            logger.warning("Missing 'answer_idx' field in QA data")
            missing_qa += 1
            continue
            
        if str(qa["pred"]) == str(qa["answer_idx"]):
            correct_count += 1
    
    valid_predictions = len(pred_data) - missing_qa
    accuracy = correct_count / len(pred_data) if len(pred_data) > 0 else 0.0
    
    return accuracy, missing_qa


def eval_mcq_skills(pred_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple-choice question skills using accuracy metrics.
    
    Args:
        pred_dir (str): Directory containing prediction files
        
    Returns:
        Dict[str, Dict[str, Any]]: Results dictionary with skill names as keys and metrics as values
    """
    skills_mcq = [
        "character_actions",
        "scene_transitions", 
        "choronological_understanding",
        "global_appearance",
    ]
    
    logger.info("Starting evaluation of MCQ skills...")
    
    if not os.path.exists(pred_dir):
        logger.error(f"Prediction directory does not exist: {pred_dir}")
        return {}
    
    skill_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    mcq_results = {}
    
    for skill_file in skill_files:
        skill_name = skill_file.split('.')[0]
        if skill_name not in skills_mcq:
            logger.debug(f"Skipping {skill_name} (not an MCQ skill)")
            continue
            
        skill_path = os.path.join(pred_dir, skill_file)
        # logger.info(f"{'='*50}")
        # logger.info(f"Processing MCQ skill: {skill_name}")
        # logger.info(f"{'='*50}")
        
        try:
            pred_data = read_json_file(skill_path)
        except Exception as e:
            logger.error(f"Failed to read {skill_path}: {e}")
            continue
        
        if not pred_data:
            logger.warning(f"No data found in {skill_path}")
            continue
        
        accuracy, missing_qa = mcq_accuracy(pred_data)
        mcq_results[skill_name] = {
            'accuracy': accuracy,
            'total_items': len(pred_data),
            'missing_predictions': missing_qa
        }
        
        if missing_qa > 0:
            logger.warning(f"{missing_qa}/{len(pred_data)} QA pairs had missing predictions")
        
        # logger.info(f"Accuracy for {skill_name}: {accuracy*100:.2f}% "
        #            f"({len(pred_data) - missing_qa}/{len(pred_data)} valid predictions)")
   
    logger.info("MCQ skills evaluation completed.")
    return mcq_results
    

def print_comprehensive_summary(open_ended_results: Dict[str, Dict[str, Any]], 
                              mcq_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a comprehensive summary of both open-ended and MCQ evaluation results.
    
    Args:
        open_ended_results: Results from open-ended skills evaluation
        mcq_results: Results from MCQ skills evaluation
    """
    logger.info("="*80)
    logger.info("🎯 INFINIBENCH EVALUATION SUMMARY")
    logger.info("="*80)
    
    # Overall evaluation status
    total_skills_evaluated = len(open_ended_results) + len(mcq_results)
    logger.info(f"📊 Total Skills Evaluated: {total_skills_evaluated}")
    logger.info(f"   • Open-ended Skills: {len(open_ended_results)}")
    logger.info(f"   • Multiple-Choice Skills: {len(mcq_results)}")
    logger.info("")
    
    # Open-ended results summary
    if open_ended_results:
        logger.info("OPEN-ENDED SKILLS PERFORMANCE")
        logger.info("-" * 50)
        
        # Calculate overall open-ended statistics
        total_valid_scores = sum(result['valid_scores'] for result in open_ended_results.values())
        total_items_all = sum(result['total_items'] for result in open_ended_results.values())
        
        if total_valid_scores > 0:
            # Calculate average score over the open-ended skills
            overall_average = sum(result['average_score'] for result in open_ended_results.values()) / len(open_ended_results)
            
            # Find overall min and max scores
            all_min_scores = [result['min_score'] for result in open_ended_results.values() if result['valid_scores'] > 0]
            all_max_scores = [result['max_score'] for result in open_ended_results.values() if result['valid_scores'] > 0]
            overall_min = min(all_min_scores) if all_min_scores else 0
            overall_max = max(all_max_scores) if all_max_scores else 0
            

            logger.info(f"Overall GPT Score: {overall_average:.2f}/10")
            logger.info(f"Score Range: {overall_min} - {overall_max}")
            logger.info(f"Success Rate: {total_valid_scores}/{total_items_all} ({total_valid_scores/total_items_all*100:.1f}%)")
            logger.info("")
            
            # Individual skill breakdown
            logger.info("Skill-by-Skill Breakdown:")
            sorted_skills = sorted(open_ended_results.items(), key=lambda x: x[1]['average_score'], reverse=True)
            for skill, result in sorted_skills:
                if result['valid_scores'] > 0:
                    success_rate = result['valid_scores'] / result['total_items'] * 100
                    logger.info(f"  {skill:<30} {result['average_score']:>5.2f}/10  "
                               f"(Success: {success_rate:>5.1f}%) Range: {result['min_score']}-{result['max_score']} "
                               f"(Valid: {result['valid_scores']}, Missing: {result['missing_scores']})")
                else:
                    logger.info(f"  {skill:<30} {'N/A':>5}     (No valid scores)")
        else:
            logger.info("❌ No valid GPT scores obtained for open-ended skills")
    else:
        logger.info("⏭️  Open-ended skills evaluation was skipped")
    
    logger.info("")
    
    # MCQ results summary
    if mcq_results:
        logger.info("MULTIPLE-CHOICE SKILLS PERFORMANCE")
        logger.info("-" * 50)
        
        # Calculate overall MCQ statistics
        total_accuracy = sum(result['accuracy'] for result in mcq_results.values()) / len(mcq_results)
        total_mcq_items = sum(result['total_items'] for result in mcq_results.values())
        total_mcq_missing = sum(result['missing_predictions'] for result in mcq_results.values())
        total_mcq_valid = total_mcq_items - total_mcq_missing
        
        # Performance classification
        
        logger.info(f"Overall Accuracy: {total_accuracy*100:.2f}%")
        logger.info(f"Total Questions: {total_mcq_items} (Valid: {total_mcq_valid}, Missing: {total_mcq_missing})")
        logger.info("")
        
        # Individual skill breakdown
        logger.info("Skill-by-Skill Breakdown:")
        sorted_mcq_skills = sorted(mcq_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for skill, result in sorted_mcq_skills:
            valid_predictions = result['total_items'] - result['missing_predictions']
            success_rate = valid_predictions / result['total_items'] * 100 if result['total_items'] > 0 else 0
            logger.info(f"  {skill:<30} {result['accuracy']*100:>5.1f}%   "
                       f"(Valid: {success_rate:>5.1f}%) Missing: {result['missing_predictions']} "
                       f"(Total: {result['total_items']})")
    else:
        logger.info("⏭️  Multiple-choice skills evaluation was skipped")
    
  
    logger.info("")
    logger.info("="*80)
    

def main() -> None:
    """Main function to orchestrate the evaluation process."""
    
    parser = argparse.ArgumentParser(
        description="InfiniBench Evaluation Script - Automated evaluation for video-based QA tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pred_dir ./predictions
  %(prog)s --pred_dir ./predictions --max_threads 8 --batch_size 10
  %(prog)s --pred_dir ./predictions --skip_open_ended
  %(prog)s --pred_dir ./predictions --skip_mcq
        """
    )
    
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        required=True, 
        help="Directory containing prediction files (JSON format)"
    )
    parser.add_argument(
        "--max_threads", 
        type=int, 
        default=4, 
        help="Maximum number of threads for parallel processing (default: 4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of items to process before saving (for safety, default: 5)"
    )
    parser.add_argument(
        "--skip_open_ended",
        action="store_true",
        help="Skip evaluation of open-ended skills"
    )
    parser.add_argument(
        "--skip_mcq",
        action="store_true",
        help="Skip evaluation of multiple-choice questions"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for failed API requests (default: 3)"
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=5.0,
        help="Base delay between retries in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Validate arguments
    if not os.path.exists(args.pred_dir):
        logger.error(f"Prediction directory does not exist: {args.pred_dir}")
        sys.exit(1)
    
    if args.max_threads < 1:
        logger.error("max_threads must be at least 1")
        sys.exit(1)
        
    if args.batch_size < 1:
        logger.error("batch_size must be at least 1")
        sys.exit(1)
    
    if args.max_retries < 0:
        logger.error("max_retries must be at least 0")
        sys.exit(1)
        
    if args.retry_delay < 0:
        logger.error("retry_delay must be non-negative")
        sys.exit(1)
    
    if args.skip_open_ended and args.skip_mcq:
        logger.error("Cannot skip both open-ended and MCQ evaluations")
        sys.exit(1)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") and not args.skip_open_ended:
        logger.error("OPENAI_API_KEY environment variable is required for open-ended evaluation")
        logger.error("Set it with: export OPENAI_API_KEY='your-key-here' (Linux/Mac) or $env:OPENAI_API_KEY='your-key-here' (Windows)")
        sys.exit(1)
    
    logger.info("*"*70)
    logger.info("INFINIBENCH EVALUATION PARAMETERS")
    logger.info("*"*70)
    logger.info(f"Prediction directory: {args.pred_dir}")
    logger.info(f"Max threads: {args.max_threads}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Retry delay: {args.retry_delay}s")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Skipping open-ended skills: {args.skip_open_ended}")
    logger.info(f"Skipping MCQ skills: {args.skip_mcq}")
    logger.info("*"*70)
    
    try:
        # Initialize result containers
        open_ended_results = {}
        mcq_results = {}
        
        # Run evaluations
        if not args.skip_open_ended:
            open_ended_results = eval_open_ended_skills(args.pred_dir, args.max_threads, args.batch_size, args.max_retries, args.retry_delay)
        else:
            logger.info("Skipping open-ended skills evaluation")
        
        logger.info("*"*40)
        if not args.skip_mcq:
            mcq_results = eval_mcq_skills(args.pred_dir)
        else:
            logger.info("Skipping MCQ skills evaluation")
        logger.info("*"*40)
        # Print comprehensive summary
        print_comprehensive_summary(open_ended_results, mcq_results)

        logger.info("Check 'evaluation.log' for detailed logs")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()