import 'dart:io';

/// Loads a BERT-style vocabulary into a `Map<String, int>` (token → ID).
///
/// BERT vocabulary files are plain text with **one token per line**. The line
/// index (0-based) is the token's integer ID. For example:
///
/// ```
/// [PAD]        ← ID 0
/// [unused1]    ← ID 1
/// ...
/// [UNK]        ← ID 100
/// [CLS]        ← ID 101
/// [SEP]        ← ID 102
/// ...
/// flutter      ← some ID
/// ##ing        ← continuation sub-word
/// ```
///
/// Three loading strategies are provided:
///
/// | Method | Source | Use case |
/// |---|---|---|
/// | [VocabLoader.fromFile] | `dart:io` [File] | Flutter / CLI apps |
/// | [VocabLoader.fromString] | Raw `String` | Assets, embedded vocabs |
/// | [VocabLoader.fromMap] | Pre-built `Map` | Tests, custom pipelines |
///
/// Example — loading from a file:
/// ```dart
/// final vocab = await VocabLoader.fromFile(File('/path/to/vocab.txt'));
/// print(vocab['[CLS]']); // 101
/// ```
///
/// Example — loading from a Flutter asset string:
/// ```dart
/// final raw = await rootBundle.loadString('assets/vocab.txt');
/// final vocab = VocabLoader.fromString(raw);
/// ```
abstract final class VocabLoader {
  VocabLoader._();

  // ---------------------------------------------------------------------------
  // Public factory methods
  // ---------------------------------------------------------------------------

  /// Reads a vocabulary file from [file] and returns the token → ID map.
  ///
  /// The file must be UTF-8 encoded. Blank lines are skipped; leading and
  /// trailing whitespace is stripped from every token.
  ///
  /// Throws [FileSystemException] if the file does not exist or cannot be read.
  ///
  /// Example:
  /// ```dart
  /// final vocab = await VocabLoader.fromFile(File('/data/vocab.txt'));
  /// ```
  static Future<Map<String, int>> fromFile(File file) async {
    final String content = await file.readAsString();
    return fromString(content);
  }

  /// Parses a vocabulary from the raw [content] string.
  ///
  /// [content] must follow the BERT vocabulary format: one token per line,
  /// line index = token ID. Blank lines are skipped.
  ///
  /// This method is synchronous and has no I/O side effects, making it
  /// suitable for use with Flutter's `rootBundle.loadString`.
  ///
  /// Example:
  /// ```dart
  /// // Flutter asset
  /// final raw = await rootBundle.loadString('assets/vocab.txt');
  /// final vocab = VocabLoader.fromString(raw);
  ///
  /// // Inline (e.g. for unit tests)
  /// final vocab = VocabLoader.fromString('[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello');
  /// ```
  static Map<String, int> fromString(String content) {
    final Map<String, int> vocab = <String, int>{};
    final List<String> lines = content.split('\n');

    for (int i = 0; i < lines.length; i++) {
      final String token = lines[i].trim();
      if (token.isNotEmpty) {
        vocab[token] = i;
      }
    }
    return vocab;
  }

  /// Wraps an already-built [map] for use with [WordPieceTokenizer].
  ///
  /// No copying is performed; [map] is used as-is. Prefer this factory in
  /// unit tests or when vocabulary is generated programmatically.
  ///
  /// Example:
  /// ```dart
  /// final vocab = VocabLoader.fromMap({
  ///   '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
  ///   'hello': 4, '##world': 5,
  /// });
  /// ```
  static Map<String, int> fromMap(Map<String, int> map) => map;

  // ---------------------------------------------------------------------------
  // Convenience helpers
  // ---------------------------------------------------------------------------

  /// Returns the size of [vocab] (number of distinct tokens).
  static int vocabSize(Map<String, int> vocab) => vocab.length;

  /// Returns `true` if [token] is present in [vocab].
  static bool contains(Map<String, int> vocab, String token) =>
      vocab.containsKey(token);
}
