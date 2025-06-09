document.addEventListener('alpine:init', () => {
    Alpine.data('leaderboardApp', () => ({
      search: '',
      sortKey: '',
      sortAsc: true,
      skillHeaders: [
        { key: 'globalAppearance', label: 'Global Appearance' },
        { key: 'sceneTransitions', label: 'Scene Transitions' },
        { key: 'characterActions', label: 'Character Actions' },
        { key: 'chronologicalUnderstanding', label: 'Chronological Understanding' },
        { key: 'summarization', label: 'Summarization' },
        { key: 'deepContextUnderstanding', label: 'Deep Context Understanding' },
        { key: 'spoilerUnderstanding', label: 'Spoiler Understanding' },
        { key: 'linkingEvents', label: 'Linking Events' }
      ],
      data: [
        { model: 'Baseline Random', frameRate: '--', globalAppearance: 19.96, sceneTransitions: 19.77, characterActions: 18.41, chronologicalUnderstanding: 36.45, avgAcc: 23.65 },
        { model: 'GPT-4o', frameRate: 450 },
        { model: 'Gemini Flash 2.0', frameRate: '1 FPS' },
        { model: 'Qwen2.5VL', frameRate: 768, globalAppearance: 33.16, sceneTransitions: 29.85, characterActions: 29.31, chronologicalUnderstanding: 45.37, summarization: 3.34, deepContextUnderstanding: 4.82, spoilerUnderstanding: 3.67, linkingEvents: 6.39, avgAcc: 34.42, avgScore: 4.56 },
        { model: 'Video-Flash', frameRate: 1000, globalAppearance: 22.01, sceneTransitions: 30.81, characterActions: 37.67, chronologicalUnderstanding: 47.58, summarization: 2.70, deepContextUnderstanding: 3.87, spoilerUnderstanding: 2.95, linkingEvents: 5.02, avgAcc: 34.52, avgScore: 3.64 },
        { model: 'LLava-Onevision', frameRate: 128, globalAppearance: 24.19, sceneTransitions: 27.83, characterActions: 25.26, chronologicalUnderstanding: 46.50, summarization: 2.00, deepContextUnderstanding: 4.09, spoilerUnderstanding: 3.31, linkingEvents: 6.14, avgAcc: 30.95, avgScore: 3.89 },
        // Add the rest of your rows here...
      ],
      sort(key) {
        if (this.sortKey === key) {
          this.sortAsc = !this.sortAsc;
        } else {
          this.sortKey = key;
          this.sortAsc = true;
        }
        this.data.sort((a, b) => {
          if (a[key] == null) return 1;
          if (b[key] == null) return -1;
          return this.sortAsc ? a[key] - b[key] : b[key] - a[key];
        });
      },
      filteredData() {
        if (!this.search) return this.data;
        return this.data.filter(row => row.model.toLowerCase().includes(this.search.toLowerCase()));
      }
    }));
  });