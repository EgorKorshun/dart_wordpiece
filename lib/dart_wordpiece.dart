/// Pure Dart BERT-compatible WordPiece tokenizer.
///
/// Converts raw text into the three integer sequences required by BERT-style
/// ONNX models: `input_ids`, `attention_mask`, and `token_type_ids`.
///
/// ## Quick start
///
/// ```dart
/// import 'package:dart_wordpiece/dart_wordpiece.dart';
///
/// final vocab = await VocabLoader.fromFile(File('/path/to/vocab.txt'));
/// final tokenizer = WordPieceTokenizer(vocab: vocab);
///
/// final output = tokenizer.encode('What is Flutter?');
/// print(output.inputIds);
/// print(output.realLength);
/// ```
library;

export 'src/special_tokens.dart';
export 'src/text_normalizer.dart';
export 'src/tokenizer_config.dart';
export 'src/tokenizer_output.dart';
export 'src/vocab_loader.dart';
export 'src/word_piece_tokenizer.dart';
