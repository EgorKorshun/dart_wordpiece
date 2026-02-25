# dart_wordpiece

A pure Dart implementation of the BERT-compatible **WordPiece tokenizer**.

Converts raw text into the three integer sequences expected by BERT-style
ONNX models — `input_ids`, `attention_mask`, and `token_type_ids` — with
**zero dependencies** and no Flutter requirement.

---

## Features

- ✅ **WordPiece algorithm** — greedy longest-match-first subword encoding
- ✅ **Single & pair encoding** — `encode()` and `encodePair()` for QA / NLI tasks
- ✅ **Batch encoding** — `encodeAll(List<String>)`
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
  dart_wordpiece: ^0.1.0
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

## API reference

### `WordPieceTokenizer`

| Member | Description |
|---|---|
| `WordPieceTokenizer({vocab, config})` | Main constructor |
| `WordPieceTokenizer.fromFile(file, {config})` | Async factory from `dart:io` File |
| `WordPieceTokenizer.fromString(content, {config})` | Sync factory from String |
| `encode(text)` → `TokenizerOutput` | Encode single sequence |
| `encodePair(textA, textB)` → `TokenizerOutput` | Encode sentence pair |
| `encodeAll(texts)` → `List<TokenizerOutput>` | Batch encode |
| `tokenize(text)` → `List<String>` | Raw token strings (no padding) |
| `tokenToId(token)` → `int?` | Look up token ID |
| `idToToken(id)` → `String?` | Look up token string (debug) |
| `vocabSize` | Number of tokens in vocabulary |

### `TokenizerOutput`

| Member | Type | Description |
|---|---|---|
| `inputIds` | `List<int>` | Vocabulary IDs |
| `attentionMask` | `List<int>` | 1 = real token, 0 = padding |
| `tokenTypeIds` | `List<int>` | 0 = segment A, 1 = segment B |
| `length` | `int` | Always equals `maxLength` |
| `realLength` | `int` | Non-padding positions |
| `inputIdsInt64` | `Int64List` | Ready for ONNX tensor |
| `attentionMaskInt64` | `Int64List` | Ready for ONNX tensor |
| `tokenTypeIdsInt64` | `Int64List` | Ready for ONNX tensor |

### `TokenizerConfig`

| Parameter | Default | Description |
|---|---|---|
| `maxLength` | `64` | Output sequence length (includes special tokens) |
| `specialTokens` | `SpecialTokens.bert()` | `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `##` |
| `stopwords` | `{}` | Words removed before tokenization |
| `normalizeText` | `true` | Lowercase + remove punctuation |

### `VocabLoader`

| Method | Description |
|---|---|
| `VocabLoader.fromFile(File)` | Async load from file |
| `VocabLoader.fromString(String)` | Sync parse from vocab text |
| `VocabLoader.fromMap(Map)` | Wrap pre-built map |

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

## Compatibility

Compatible with vocabularies from:
- `bert-base-uncased`
- `bert-base-cased`
- `distilbert-base-uncased`
- Any model that follows the HuggingFace `vocab.txt` format

---

## License

MIT
