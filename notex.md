```bash
┌─────────────────────────────────────────────────────────┐
│ USER QUERY                                              │
│ "um, who is Tanvir Ahmed basically?"                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 🔧 QUERY OPTIMIZATION NODE                             │
│ - Clean: "who is tanvir ahmed"                         │
│ - Entities: {PERSON: ["Tanvir Ahmed"]}                │
│ - Keywords: ["tanvir ahmed", "ai engineer"]            │
│ - Intent: "factual" → Always retrieve                  │
│ - Variations: [3 alternatives]                         │
│ - Status: ✅ success                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 🧠 SEMANTIC CACHE CHECK                                │
│ - Check if similar question cached                     │
│ - If YES → Return cached answer (⚡ fast!)            │
│ - If NO → Continue to optimization                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 🤔 SMART RETRIEVAL DECISION                            │
│ - Intent: "factual" → Retrieve = TRUE                  │
│ - Use multi-query retrieval (3 queries)                │
│ - Quality check score                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
    ┌─────────────┴──────────────┐
    │                            │
    ▼                            ▼
┌──────────────────┐    ┌──────────────────┐
│ 📚 RETRIEVAL     │    │ 💡 FALLBACK      │
│ Multi-query:     │    │ Generate w/tools │
│ - Query 1        │    │ (no retrieval)   │
│ - Query 2        │    │ Status: partial  │
│ - Query 3        │    └──────┬───────────┘
│ Result: 6-8 docs │           │
└────────┬─────────┘           │
         │                     │
         └─────────┬───────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 💬 GENERATION                                          │
│ Generate answer from context                           │
│ Status: ✅ success                                    │
│ "Tanvir Ahmed is an AI Engineer..."                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ ✅ EVALUATION                                          │
│ - Support check: "fully_supported"                     │
│ - Usefulness: "useful"                                 │
│ - Revision attempts: 0 (max 2)                         │
│ - Status: success (0.92/1.0 score)                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 💾 SAVE TO SEMANTIC CACHE                              │
│ - Question embedding stored                            │
│ - Answer cached for future queries                     │
│ - Enables 60%+ cache hit rate                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ ✅ FINAL ANSWER                                        │
│ Status: success                                        │
│ Latency: 1.2s                                          │
│ Cache Hit: No (first query)                            │
│ Evaluation Score: 0.92/1.0                             │
└─────────────────────────────────────────────────────────┘
```