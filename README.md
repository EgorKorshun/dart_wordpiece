# dart_wordpiece

A pure Dart implementation of the BERT-compatible **WordPiece tokenizer**.

Converts raw text into the three integer sequences expected by BERT-style
ONNX models — `input_ids`, `attention_mask`, and `token_type_ids` — with
**zero dependencies** and no Flutter requirement.

---

## Features

- ✅ **WordPiece algorithm** — greedy longest-match-first subword encoding
- ✅ **Single & pair encoding** — `encode()` and `encodePair()` for QA / NLI tasks
- ✅ **Batch encoding** — `encodeAll()` with `PaddingStrategy.longest` support
- ✅ **Offset mapping** — character spans per token for NER and span extraction
- ✅ **Truncation control** — `TruncationSide.left` or `right`
- ✅ **Async variants** — `encodeAsync` / `encodeAllAsync` for Flutter isolates
- ✅ **Text normalization** — lowercase, punctuation removal, stopword filtering
- ✅ **Three vocab loading strategies** — from `File`, from `String`, from `Map`
- ✅ **Configurable** — max length, special tokens, stopwords, normalization toggle
- ✅ **ONNX-ready** — `Int64List` getters for direct tensor creation
- ✅ **Pure Dart** — works in Flutter, CLI, and server-side Dart

---

## Getting started

Add to your `pubspec.yaml`:

```yaml
dependencies:
  dart_wordpiece: ^1.2.0
```

---

## Usage

### 1. Load vocabulary

```dart
import 'package:dart_wordpiece/dart_wordpiece.dart';

// From a file (dart:io)
final vocab = await VocabLoader.fromFile(File('/path/to/vocab.txt'));

// From a Flutter asset string
final raw = await rootBundle.loadString('assets/vocab.txt');
final vocab = VocabLoader.fromString(raw);

// From an in-memory map (useful for tests)
final vocab = VocabLoader.fromMap({'[PAD]': 0, '[UNK]': 1, ...});
```

### 2. Create tokenizer

```dart
// Default BERT configuration (maxLength=64, no stopwords)
final tokenizer = WordPieceTokenizer(vocab: vocab);

// Custom configuration
final tokenizer = WordPieceTokenizer(
  vocab: vocab,
  config: TokenizerConfig(
    maxLength: 128,
    stopwords: {'what', 'is', 'the', 'a', 'an'},
    normalizeText: true,   // lowercase + remove punctuation
  ),
);
```

### 3. Encode a single sequence

```dart
final output = tokenizer.encode('What is Flutter?');

print(output.inputIds);      // [101, 2054, 2003, 14246, 2102, 1029, 102, 0, …]
print(output.attentionMask); // [1, 1, 1, 1, 1, 1, 1, 0, …]
print(output.tokenTypeIds);  // [0, 0, 0, 0, 0, 0, 0, 0, …]
print(output.realLength);    // 7  (non-padding positions)
```

### 4. Encode a sentence pair

Use for BERT-based **question answering** or **natural language inference**:

```dart
final output = tokenizer.encodePair(
  'Flutter is a cross-platform UI toolkit.',  // segment A
  'What is Flutter?',                          // segment B
);

// Format:  [CLS] <A> [SEP] <B> [SEP] [PAD]…
// typeIds:   0    0    0    1    1    0…
```

### 5. Batch encoding

```dart
final outputs = tokenizer.encodeAll([
  'What is Flutter?',
  'Dart is fast',
  'How to use isolates?',
]);
// Each output has the same length (maxLength) → stack into a batch tensor.
```

### 6. Inspect raw token strings

```dart
tokenizer.tokenize('unaffable');
// → ['[CLS]', 'un', '##aff', '##able', '[SEP]']
```

### 7. Feed directly to an ONNX model

```dart
// package:onnxruntime integration
final out = tokenizer.encode(query);
final inputs = {
  'input_ids':      OrtValueTensor.createTensorWithDataList(out.inputIdsInt64,       [1, out.length]),
  'attention_mask': OrtValueTensor.createTensorWithDataList(out.attentionMaskInt64,  [1, out.length]),
  'token_type_ids': OrtValueTensor.createTensorWithDataList(out.tokenTypeIdsInt64,   [1, out.length]),
};
final results = session.run(OrtRunOptions(), inputs);
```

---

## New in v1.2.0

### Offset mapping

`TokenizerOutput.offsetMapping` returns a `List<(int, int)>` of `(start, end)` character spans in the normalized text for each token position. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) get the sentinel `(0, 0)`.

Useful for **NER**, **QA span extraction**, and **text highlighting** in Flutter:

```dart
final output = tokenizer.encode('playing dart');

// Tokens:  [CLS]  play  ##ing  dart  [SEP]  [PAD]…
// Offsets: (0,0) (0,4) (4,7) (8,12) (0,0) (0,0)…
print(output.offsetMapping);
// → [(0,0), (0,4), (4,7), (8,12), (0,0), ...]

// Example: highlight predicted span in original text
final spanStart = output.offsetMapping![tokenStart].$1;
final spanEnd   = output.offsetMapping![tokenEnd].$2;
final highlight = normalizedText.substring(spanStart, spanEnd);
```

### Special tokens mask

`TokenizerOutput.specialTokensMask` is `1` for `[CLS]`/`[SEP]`/`[PAD]` and `0` for real content tokens. Useful for masked language model post-processing:

```dart
print(output.specialTokensMask);
// encode('flutter is fast')  →  [1, 0, 0, 0, 1, 1, 1, …]
//                                CLS  ↑         SEP PAD
```

### Overflowing tokens

`TokenizerOutput.overflowingTokens` lists tokens that were removed by truncation, in order. Foundation for sliding-window approaches over long documents:

```dart
final config = TokenizerConfig(maxLength: 8);
final output = tokenizer.encode(veryLongText);
if (output.overflowingTokens!.isNotEmpty) {
  print('Truncated: ${output.overflowingTokens}');
}
```

### Padding strategy

`PaddingStrategy.longest` pads the batch to the length of its longest sequence instead of `maxLength`. Saves compute when sequences vary in length:

```dart
final tokenizer = WordPieceTokenizer(
  vocab: vocab,
  config: TokenizerConfig(
    maxLength: 512,
    paddingStrategy: PaddingStrategy.longest,
  ),
);

final outputs = tokenizer.encodeAll(shortTexts);
// outputs[i].length == longest sequence in batch (not 512)
```

### Truncation side

`TruncationSide.left` keeps the **last** N tokens instead of the first. Useful for models that attend to document endings:

```dart
final tokenizer = WordPieceTokenizer(
  vocab: vocab,
  config: TokenizerConfig(
    maxLength: 64,
    truncationSide: TruncationSide.left,
  ),
);
```

### Async variants

`encodeAsync`, `encodePairAsync`, and `encodeAllAsync` integrate with Flutter's `compute()` and `Isolate.run()`:

```dart
// Returns a Future — safe to await in async Flutter code.
final output = await tokenizer.encodeAsync(text);

// encodeAllAsync offloads to a separate isolate for CPU-intensive batches.
final outputs = await tokenizer.encodeAllAsync(largeTextList);
```

---

## API reference

### `WordPieceTokenizer`

| Member                                                      | Description                           |
|-------------------------------------------------------------|---------------------------------------|
| `WordPieceTokenizer({vocab, config})`                       | Main constructor                      |
| `WordPieceTokenizer.fromFile(file, {config})`               | Async factory from `dart:io` File     |
| `WordPieceTokenizer.fromString(content, {config})`          | Sync factory from String              |
| `encode(text)` → `TokenizerOutput`                          | Encode single sequence                |
| `encodePair(textA, textB)` → `TokenizerOutput`              | Encode sentence pair                  |
| `encodeAll(texts)` → `List<TokenizerOutput>`                | Batch encode                          |
| `encodeAsync(text)` → `Future<TokenizerOutput>`             | Async encode                          |
| `encodePairAsync(textA, textB)` → `Future<TokenizerOutput>` | Async pair encode                     |
| `encodeAllAsync(texts)` → `Future<List<TokenizerOutput>>`   | Async batch encode (separate isolate) |
| `tokenize(text)` → `List<String>`                           | Raw token strings (no padding)        |
| `tokenToId(token)` → `int?`                                 | Look up token ID                      |
| `idToToken(id)` → `String?`                                 | Look up token string — O(1)           |
| `vocabSize`                                                 | Number of tokens in vocabulary        |

### `TokenizerOutput`

| Member               | Type               | Description                             |
|----------------------|--------------------|-----------------------------------------|
| `inputIds`           | `List<int>`        | Vocabulary IDs                          |
| `attentionMask`      | `List<int>`        | 1 = real token, 0 = padding             |
| `tokenTypeIds`       | `List<int>`        | 0 = segment A, 1 = segment B            |
| `offsetMapping`      | `List<(int,int)>?` | Char spans per token in normalized text |
| `specialTokensMask`  | `List<int>?`       | 1 = [CLS]/[SEP]/[PAD], 0 = content      |
| `overflowingTokens`  | `List<String>?`    | Tokens removed by truncation            |
| `length`             | `int`              | Total token slots                       |
| `realLength`         | `int`              | Non-padding positions                   |
| `inputIdsInt64`      | `Int64List`        | Ready for ONNX tensor                   |
| `attentionMaskInt64` | `Int64List`        | Ready for ONNX tensor                   |
| `tokenTypeIdsInt64`  | `Int64List`        | Ready for ONNX tensor                   |

### `TokenizerConfig`

| Parameter         | Default                 | Description                                                    |
|-------------------|-------------------------|----------------------------------------------------------------|
| `maxLength`       | `64`                    | Output sequence length (includes special tokens)               |
| `specialTokens`   | `SpecialTokens.bert()`  | `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `##`                       |
| `stopwords`       | `{}`                    | Words removed before tokenization                              |
| `normalizeText`   | `true`                  | Lowercase + remove punctuation                                 |
| `paddingStrategy` | `PaddingStrategy.fixed` | `fixed` = pad to `maxLength`; `longest` = pad to batch longest |
| `truncationSide`  | `TruncationSide.right`  | `right` = drop from end; `left` = drop from start              |

### `VocabLoader`

| Method                           | Description                |
|----------------------------------|----------------------------|
| `VocabLoader.fromFile(File)`     | Async load from file       |
| `VocabLoader.fromString(String)` | Sync parse from vocab text |
| `VocabLoader.fromMap(Map)`       | Wrap pre-built map         |

---

## WordPiece algorithm

1. Split text into whitespace-delimited words.
2. For each word, find the **longest prefix** present in the vocabulary.
3. Emit the prefix as a token; prepend `##` to the remaining suffix and repeat.
4. If no single-character match exists, emit `[UNK]` for the whole word.

```
"unaffable"  →  ["un", "##aff", "##able"]
"playing"    →  ["play", "##ing"]
"xyz123"     →  ["[UNK]"]   (if none of the sub-strings are in vocab)
```

---

## Where to get `vocab.txt`

Download pre-trained BERT vocabularies from HuggingFace:

| Model             | Link                                                                                 | Language        | Size         |
|-------------------|--------------------------------------------------------------------------------------|-----------------|--------------|
| BERT Base Uncased | [vocab.txt](https://huggingface.co/bert-base-uncased/blob/main/vocab.txt)            | English         | ~100K tokens |
| BERT Base Cased   | [vocab.txt](https://huggingface.co/bert-base-cased/blob/main/vocab.txt)              | English (cased) | ~28K tokens  |
| BERT Multilingual | [vocab.txt](https://huggingface.co/bert-base-multilingual-cased/blob/main/vocab.txt) | 104 languages   | ~120K tokens |
| RuBERT (Russian)  | [vocab.txt](https://huggingface.co/cointegrated/rubert-tiny2/blob/main/vocab.txt)    | Russian         | ~119K tokens |
| DistilBERT        | [vocab.txt](https://huggingface.co/distilbert-base-uncased/blob/main/vocab.txt)      | English         | ~30K tokens  |

**Quick download** (in a terminal):
```bash
curl -o vocab.txt https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
```

For Flutter apps, place `vocab.txt` in `assets/` and update `pubspec.yaml`:
```yaml
flutter:
  assets:
    - assets/vocab.txt
```

---

## Multilingual support

The tokenizer works with **any language** — just use a multilingual vocab:

```dart
// Load multilingual BERT vocab
final raw = await rootBundle.loadString('assets/multilingual_vocab.txt');
final vocab = VocabLoader.fromString(raw);
final tokenizer = WordPieceTokenizer(vocab: vocab);

// Works with English, Chinese, Russian, Arabic, etc.
tokenizer.encode('Hello World');        // English
tokenizer.encode('你好世界');           // Chinese
tokenizer.encode('Привет мир');        // Russian
tokenizer.encode('مرحبا بالعالم');     // Arabic
```

The `TextNormalizer` uses Unicode-aware regex (`\p{L}`, `\p{N}`), so punctuation removal and letter detection work correctly for all writing systems.

---

## Compatibility

Compatible with vocabularies from:
- `bert-base-uncased`
- `bert-base-cased`
- `distilbert-base-uncased`
- `bert-base-multilingual-cased`
- `cointegrated/rubert-tiny2`
- Any model that follows the HuggingFace `vocab.txt` format

---

## Performance

- **Single tokenization**: ~1–5 ms (on modern hardware)
- **Batch of 100 queries**: ~50–200 ms
- **Memory footprint**: Vocab size × ~8 bytes (e.g., 30K tokens = ~240 KB)
- **No external dependencies** — pure Dart with optimized regex patterns

---

## Contributing

Issues and pull requests are welcome! Please:
1. Run `dart analyze` and ensure no warnings
2. Run `dart test` and ensure all tests pass
3. Follow the [Dart style guide](https://dart.dev/guides/language/effective-dart/style)

---

## License

MIT
