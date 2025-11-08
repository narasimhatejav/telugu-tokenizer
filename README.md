# Telugu BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer trained from scratch on Telugu Wikipedia articles.

## 📊 Model Metrics

- **Vocabulary Size**: **5,500 tokens** ✅
- **Compression Ratio**: **11.80x** ✅
- **Training Corpus**: 300 Telugu Wikipedia articles
- **Corpus Size**: 1.1 million characters (2.87 million bytes)
- **Base Tokens**: 256 (UTF-8 bytes)
- **Learned Merges**: 5,244



## 📈 Compression Performance

The tokenizer achieves excellent compression on Telugu text:

| Metric | Value |
|--------|-------|
| Original bytes | 2,871,488 |
| Compressed tokens | 243,272 |
| Compression ratio | **11.80x** |
| Average bytes per token | 11.80 |

### Compression Progression

During training, compression improved progressively:

- After 100 merges: 3.59x
- After 500 merges: 5.64x
- After 1,000 merges: 6.86x
- After 2,500 merges: 9.43x
- After 5,000 merges: 11.65x
- **Final (5,244 merges): 11.80x**

## 🧪 Example Tokenizations

### Example 1: తెలుగు (Telugu)
- **Tokens**: `[287, 2947, 264]`
- **Token count**: 3
- **Compression**: ~3.67x

### Example 2: నమస్కారం (Hello)
- **Tokens**: `[271, 282, 2551, 265]`
- **Token count**: 4
- **Compression**: ~5.25x

### Example 3: తెలుగు భాష చాలా అందంగా ఉంది
- **Translation**: "Telugu language is very beautiful"
- **Tokens**: `[287, 2947, 275, 2497, 2679, 300, 929, 443, 594, 261]`
- **Token count**: 10
- **Compression**: ~5.10x

## 🏗️ Architecture

### BPE Algorithm

The tokenizer uses the Byte Pair Encoding algorithm:

1. **Initialization**: Start with 256 base tokens (UTF-8 bytes 0-255)
2. **Iterative Merging**:
   - Find the most frequent consecutive byte pair
   - Replace all occurrences with a new token
   - Repeat 5,244 times
3. **Result**: 5,500 total tokens (256 + 5,244)


## 📚 Training Data

### Data Collection

- **Source**: Telugu Wikipedia
- **Articles scraped**: 300
- **Collection method**: BeautifulSoup web scraping
- **Total words**: ~140,000
- **Total characters**: 1,109,720
- **Total bytes**: 2,871,488

### Data Quality

The corpus contains diverse Telugu content including:
- Geographic articles (villages, cities)
- Cultural and historical content
- Administrative and demographic information
- Various writing styles and vocabulary

## 🎨 HuggingFace Spaces Demo

Try the interactive demo: [Telugu BPE Tokenizer on HuggingFace Spaces](https://huggingface.co/spaces/YOUR_USERNAME/telugu-bpe-tokenizer)
