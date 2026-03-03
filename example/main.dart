// ignore_for_file: avoid_print

/// dart_wordpiece — usage examples.
///
/// Run with: dart example/main.dart
library;

import 'package:dart_wordpiece/dart_wordpiece.dart';

void main() {
  // -------------------------------------------------------------------------
  // 1. Build a vocabulary inline (for demo purposes).
  //    In a real app use VocabLoader.fromFile or VocabLoader.fromString.
  // -------------------------------------------------------------------------
  final vocab = VocabLoader.fromMap({
    '[PAD]': 0,
    '[UNK]': 1,
    '[CLS]': 2,
    '[SEP]': 3,
    'flutter': 4,
    'is': 5,
    'a': 6,
    'ui': 7,
    'tool': 8,
    '##kit': 9,
    'what': 10,
    'dart': 11,
    '##lang': 12,
    'play': 13,
    '##ing': 14,
  });

  // -------------------------------------------------------------------------
  // 2. Create tokenizer with default BERT configuration.
  // -------------------------------------------------------------------------
  final tokenizer = WordPieceTokenizer(vocab: vocab);

  print('=== Example 1: basic encode ===');
  final output = tokenizer.encode('What is Flutter?');
  print('inputIds      : ${output.inputIds}');
  print('attentionMask : ${output.attentionMask}');
  print('tokenTypeIds  : ${output.tokenTypeIds}');
  print('realLength    : ${output.realLength}');
  print('');

  // -------------------------------------------------------------------------
  // 3. Show raw token strings (without padding).
  // -------------------------------------------------------------------------
  print('=== Example 2: tokenize (token strings) ===');
  print(tokenizer.tokenize('Flutter is a UI toolkit'));
  // → [CLS], flutter, is, a, ui, tool, ##kit, [SEP]
  print('');

  // -------------------------------------------------------------------------
  // 4. Sentence-pair encoding (BERT QA / NLI format).
  // -------------------------------------------------------------------------
  print('=== Example 3: encodePair ===');
  final pair = tokenizer.encodePair(
    'Flutter is a UI toolkit.',
    'What is Flutter?',
  );
  print('inputIds      : ${pair.inputIds.sublist(0, pair.realLength)}');
  print('tokenTypeIds  : ${pair.tokenTypeIds.sublist(0, pair.realLength)}');
  // Segment A → type 0, segment B → type 1
  print('');

  // -------------------------------------------------------------------------
  // 5. Batch encoding.
  // -------------------------------------------------------------------------
  print('=== Example 4: encodeAll ===');
  final batch = tokenizer.encodeAll([
    'What is Flutter?',
    'Dart is fast',
    'playing dart',
  ]);
  for (final o in batch) {
    print('ids=${o.inputIds.sublist(0, o.realLength)}  len=${o.realLength}');
  }
  print('');

  // -------------------------------------------------------------------------
  // 6. Stopword filtering (matches training pipeline).
  // -------------------------------------------------------------------------
  print('=== Example 5: stopword filtering ===');
  final tFiltered = WordPieceTokenizer(
    vocab: vocab,
    config: const TokenizerConfig(
      stopwords: {'what', 'is', 'a'},
    ),
  );
  print(tFiltered.tokenize('What is Flutter?'));
  // → [CLS], flutter, [SEP]   ("what", "is" removed)
  print('');

  // -------------------------------------------------------------------------
  // 7. Custom maxLength.
  // -------------------------------------------------------------------------
  print('=== Example 6: custom maxLength=6 ===');
  final tShort = WordPieceTokenizer(
    vocab: vocab,
    config: const TokenizerConfig(maxLength: 6),
  );
  final short = tShort.encode('Flutter is a UI toolkit');
  print('inputIds (len=${short.length}): ${short.inputIds}');
  print('');

  // -------------------------------------------------------------------------
  // 8. TextNormalizer standalone.
  // -------------------------------------------------------------------------
  print('=== Example 7: TextNormalizer standalone ===');
  const normalizer = TextNormalizer(stopWords: {'what', 'is', 'the'});
  print(normalizer.normalize('What IS the Dart language?'));
  // → 'dart language'
  print(normalizer.contentWords('How does Flutter handle state?'));
  // → ['does', 'flutter', 'handle', 'state']
  print('');

  // -------------------------------------------------------------------------
  // 9. Token ↔ ID conversions (useful for debugging and post-processing).
  // -------------------------------------------------------------------------
  print('=== Example 8: tokenToId / idToToken ===');
  print(tokenizer.tokenToId('[CLS]')); // → 2
  print(tokenizer.tokenToId('flutter')); // → 4
  print(tokenizer.idToToken(9)); // → '##kit'
  print(tokenizer.idToToken(999)); // → null (unknown id)
  print('vocabSize: ${tokenizer.vocabSize}');
  print('');

  // -------------------------------------------------------------------------
  // 10. Int64List tensors for ONNX Runtime.
  // -------------------------------------------------------------------------
  print('=== Example 9: Int64List for ONNX ===');
  final onnxOut = tokenizer.encode('Flutter is a UI toolkit');
  print('inputIds (Int64List) : ${onnxOut.inputIdsInt64}');
  print('attentionMask        : ${onnxOut.attentionMaskInt64}');
  print('tokenTypeIds         : ${onnxOut.tokenTypeIdsInt64}');
}
