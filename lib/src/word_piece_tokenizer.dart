import 'dart:io';

import 'special_tokens.dart';
import 'text_normalizer.dart';
import 'tokenizer_config.dart';
import 'tokenizer_output.dart';
import 'vocab_loader.dart';

/// A BERT-compatible WordPiece tokenizer implemented in pure Dart.
///
/// Converts raw text into the three integer sequences expected by BERT-style
/// models: `input_ids`, `attention_mask`, and `token_type_ids`.
///
/// ## WordPiece algorithm
///
/// WordPiece is a greedy, longest-match-first subword tokenization algorithm:
///
/// 1. The input text is split into whitespace-delimited words.
/// 2. For each word the algorithm tries to find the longest prefix present in
///    the vocabulary.
/// 3. The matched prefix is emitted as a token; the unmatched suffix is
///    prefixed with [SpecialTokens.subwordPrefix] (e.g. `##`) and the search
///    repeats on the remainder.
/// 4. If no prefix of length ≥ 1 matches, the entire remaining string is
///    replaced with [SpecialTokens.unk].
///
/// Example: `"unaffable"` → `["un", "##aff", "##able"]`
///
/// ## Usage
///
/// ```dart
/// // 1. Load vocabulary (three strategies available via VocabLoader).
/// final vocab = await VocabLoader.fromFile(File('/path/to/vocab.txt'));
///
/// // 2. Create tokenizer.
/// final tokenizer = WordPieceTokenizer(vocab: vocab);
///
/// // 3. Encode a single sequence.
/// final output = tokenizer.encode('What is Flutter?');
/// print(output.inputIds);      // [101, 2054, 2003, 14246, 2102, 1029, 102, 0, …]
/// print(output.attentionMask); // [1, 1, 1, 1, 1, 1, 1, 0, …]
/// print(output.realLength);    // 7
///
/// // 4. Encode a sentence pair (BERT QA / NLI).
/// final pair = tokenizer.encodePair('Flutter is a UI toolkit.', 'What is Flutter?');
/// // tokenTypeIds: [0,0,0,0,0,0,0, 1,1,1,1,1,1, 0,0,…]
///
/// // 5. Batch encoding.
/// final batch = tokenizer.encodeAll(['hello world', 'foo bar']);
/// ```
///
/// ## Notes
///
/// - The tokenizer is **stateless** after construction; all methods are safe to
///   call from multiple isolates simultaneously.
/// - Input text is always lowercased internally before tokenization regardless
///   of vocabulary casing, because BERT vocabularies are uncased by default.
///   Set [TokenizerConfig.normalizeText] to `false` for cased vocabularies.
class WordPieceTokenizer {
  /// Creates a [WordPieceTokenizer] from a pre-loaded [vocab] map.
  ///
  /// [vocab] maps token strings to their integer IDs (e.g. `{'[CLS]': 101}`).
  /// Build it with [VocabLoader.fromFile], [VocabLoader.fromString], or
  /// [VocabLoader.fromMap].
  ///
  /// [config] controls sequence length, special tokens, and normalization.
  /// Defaults to standard BERT settings if omitted.
  ///
  /// Throws [ArgumentError] if any required special token is absent from [vocab].
  WordPieceTokenizer({
    required Map<String, int> vocab,
    TokenizerConfig config = const TokenizerConfig(),
  })  : _vocab = vocab,
        _reverseVocab = {for (final e in vocab.entries) e.value: e.key},
        _config = config,
        _normalizer = TextNormalizer(stopWords: config.stopwords) {
    _validateSpecialTokens();
  }

  // ---------------------------------------------------------------------------
  // Named constructors
  // ---------------------------------------------------------------------------

  /// Creates a [WordPieceTokenizer] by reading a vocabulary file from [file].
  ///
  /// Shorthand for:
  /// ```dart
  /// final vocab = await VocabLoader.fromFile(file);
  /// final tokenizer = WordPieceTokenizer(vocab: vocab, config: config);
  /// ```
  static Future<WordPieceTokenizer> fromFile(
    File file, {
    TokenizerConfig config = const TokenizerConfig(),
  }) async {
    final Map<String, int> vocab = await VocabLoader.fromFile(file);
    return WordPieceTokenizer(vocab: vocab, config: config);
  }

  /// Creates a [WordPieceTokenizer] by parsing a vocabulary string.
  ///
  /// [content] must follow the BERT vocabulary format (one token per line).
  /// Useful with Flutter's `rootBundle.loadString`.
  ///
  /// ```dart
  /// final raw = await rootBundle.loadString('assets/vocab.txt');
  /// final tokenizer = WordPieceTokenizer.fromString(raw);
  /// ```
  factory WordPieceTokenizer.fromString(
    String content, {
    TokenizerConfig config = const TokenizerConfig(),
  }) {
    final Map<String, int> vocab = VocabLoader.fromString(content);
    return WordPieceTokenizer(vocab: vocab, config: config);
  }

  // ---------------------------------------------------------------------------
  // Private fields
  // ---------------------------------------------------------------------------

  final Map<String, int> _vocab;
  final Map<int, String> _reverseVocab;
  final TokenizerConfig _config;
  final TextNormalizer _normalizer;

  // Convenience accessors.
  SpecialTokens get _tokens => _config.specialTokens;
  int get _maxLength => _config.maxLength;

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Number of tokens in the vocabulary.
  int get vocabSize => _vocab.length;

  /// The active tokenizer configuration.
  TokenizerConfig get config => _config;

  /// Encodes a single [text] sequence into [TokenizerOutput].
  ///
  /// The output has exactly [TokenizerConfig.maxLength] positions:
  /// - Position 0: [SpecialTokens.cls]
  /// - Positions 1…n: content tokens
  /// - Position n+1: [SpecialTokens.sep]
  /// - Positions n+2…maxLength-1: [SpecialTokens.pad] (attentionMask = 0)
  ///
  /// Long sequences are truncated so that the last real token is always
  /// [SpecialTokens.sep].
  ///
  /// Example:
  /// ```dart
  /// final out = tokenizer.encode('Flutter is fast');
  /// print(out.inputIds);      // [101, …content…, 102, 0, 0, …]
  /// print(out.realLength);    // number of non-pad tokens
  /// ```
  TokenizerOutput encode(String text) {
    final List<int> contentIds = _tokenizeToIds(text);
    return _buildOutput(segmentA: contentIds);
  }

  /// Encodes a sentence pair [textA] and [textB] into a single [TokenizerOutput].
  ///
  /// The format follows BERT's two-segment encoding:
  /// ```
  /// [CLS] <A tokens> [SEP] <B tokens> [SEP] [PAD] …
  /// token_type_ids: 0 0…0 0 1…1 1 0…0
  /// ```
  ///
  /// Use this for tasks that require two inputs, such as:
  /// - Question answering (context + question)
  /// - Natural Language Inference (premise + hypothesis)
  /// - Sentence similarity
  ///
  /// When the combined sequence exceeds [TokenizerConfig.maxLength], segment B
  /// is truncated first (the context / longer segment should be [textA]).
  ///
  /// Example:
  /// ```dart
  /// final out = tokenizer.encodePair(
  ///   'Flutter is a UI toolkit.',
  ///   'What is Flutter?',
  /// );
  /// ```
  TokenizerOutput encodePair(String textA, String textB) {
    final List<int> idsA = _tokenizeToIds(textA);
    final List<int> idsB = _tokenizeToIds(textB);
    return _buildOutput(segmentA: idsA, segmentB: idsB);
  }

  /// Encodes every string in [texts] and returns a list of [TokenizerOutput].
  ///
  /// Each output has the same length ([TokenizerConfig.maxLength]), making it
  /// straightforward to stack into a batch tensor.
  ///
  /// Example:
  /// ```dart
  /// final outputs = tokenizer.encodeAll(['hello', 'world']);
  /// ```
  List<TokenizerOutput> encodeAll(List<String> texts) =>
      texts.map(encode).toList();

  /// Runs the WordPiece algorithm and returns token strings (not IDs).
  ///
  /// Unlike [encode], this method does **not** add special tokens or pad the
  /// output. It is useful for debugging, visualising tokenization, or building
  /// custom pipelines.
  ///
  /// Example:
  /// ```dart
  /// tokenizer.tokenize('unaffable');
  /// // → ['[CLS]', 'un', '##aff', '##able', '[SEP]']
  /// ```
  List<String> tokenize(String text) {
    final String input =
        _config.normalizeText ? _normalizer.normalize(text) : text.toLowerCase();

    final List<String> tokenStrings = <String>[_tokens.cls];
    _wordpieceEncode(input, tokenStrings);
    tokenStrings.add(_tokens.sep);
    return tokenStrings;
  }

  /// Converts a token string back to its vocabulary ID.
  ///
  /// Returns `null` if [token] is not in the vocabulary.
  int? tokenToId(String token) => _vocab[token];

  /// Converts a vocabulary ID back to its token string.
  ///
  /// Returns `null` if [id] has no corresponding token.
  String? idToToken(int id) => _reverseVocab[id];

  // ---------------------------------------------------------------------------
  // Core WordPiece algorithm
  // ---------------------------------------------------------------------------

  /// Applies the WordPiece algorithm to [text] and appends token strings to
  /// [out].
  ///
  /// The algorithm processes each whitespace-delimited word independently:
  ///
  /// 1. Try to find the longest prefix of [remaining] in the vocabulary.
  ///    - For the first piece of a word: look up [piece] directly.
  ///    - For continuation pieces: look up `##[piece]`.
  /// 2. If a match is found, emit the token and repeat on the suffix.
  /// 3. If no match is found (not even a single character), emit [SpecialTokens.unk]
  ///    and skip the rest of the word.
  void _wordpieceEncode(String text, List<String> out) {
    final List<String> words = text.split(RegExp(r'\s+'));

    for (final String word in words) {
      if (word.isEmpty) continue;

      String remaining = word;
      bool isFirstPiece = true;

      while (remaining.isNotEmpty) {
        bool matched = false;

        // Greedy longest-match-first search from remaining.length down to 1.
        for (int end = remaining.length; end > 0; end--) {
          final String candidate = remaining.substring(0, end);
          final String key =
              isFirstPiece ? candidate : '${_tokens.subwordPrefix}$candidate';

          if (_vocab.containsKey(key)) {
            out.add(key);
            remaining = remaining.substring(end);
            isFirstPiece = false;
            matched = true;
            break;
          }
        }

        if (!matched) {
          // No vocabulary match for any prefix → replace whole word with UNK.
          out.add(_tokens.unk);
          break;
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Encoding helpers
  // ---------------------------------------------------------------------------

  /// Normalizes [text] and returns a list of vocabulary IDs (without special
  /// tokens or padding).
  List<int> _tokenizeToIds(String text) {
    final String input =
        _config.normalizeText ? _normalizer.normalize(text) : text.toLowerCase();
    final List<String> pieces = <String>[];
    _wordpieceEncode(input, pieces);

    final int unkId = _vocab[_tokens.unk]!;
    return pieces.map((String t) => _vocab[t] ?? unkId).toList();
  }

  /// Assembles the final [TokenizerOutput] from pre-tokenized segment IDs.
  ///
  /// Single-segment: `[CLS] segA [SEP] [PAD]…`
  /// Two-segment:    `[CLS] segA [SEP] segB [SEP] [PAD]…`
  ///
  /// Truncates [segmentB] first when the total exceeds [_maxLength].
  TokenizerOutput _buildOutput({
    required List<int> segmentA,
    List<int>? segmentB,
  }) {
    final int clsId = _vocab[_tokens.cls]!;
    final int sepId = _vocab[_tokens.sep]!;
    final int padId = _vocab[_tokens.pad]!;

    // Reserve slots: 1 CLS + 1 SEP for segment A (+ 1 SEP for segment B).
    final int reservedSlots = segmentB == null ? 2 : 3;
    final int available = _maxLength - reservedSlots;

    List<int> a = segmentA;
    List<int> b = segmentB ?? <int>[];

    if (a.length + b.length > available) {
      // Truncate segment B first; then segment A if still too long.
      final int bAllowed = (available - a.length).clamp(0, b.length);
      b = b.sublist(0, bAllowed);
      final int aAllowed = (available - b.length).clamp(0, a.length);
      a = a.sublist(0, aAllowed);
    }

    // Build flat token ID list.
    final List<int> ids = <int>[clsId, ...a, sepId, ...b];
    if (segmentB != null) ids.add(sepId);

    // Build token_type_ids: 0 for CLS + segA + SEP, 1 for segB + SEP.
    final List<int> typeIds = <int>[
      ...List<int>.filled(a.length + 2, 0), // CLS + segA + SEP
      ...List<int>.filled(b.length + (segmentB != null ? 1 : 0), 1),
    ];

    // Attention mask: 1 for real tokens (growable so padding can be appended).
    final int realCount = ids.length;
    final List<int> mask = List<int>.filled(realCount, 1, growable: true);

    // Pad all lists to maxLength.
    while (ids.length < _maxLength) {
      ids.add(padId);
      mask.add(0);
      typeIds.add(0);
    }

    return TokenizerOutput(
      inputIds: ids,
      attentionMask: mask,
      tokenTypeIds: typeIds,
    );
  }

  // ---------------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------------

  /// Ensures all required special tokens exist in the vocabulary.
  ///
  /// Throws [ArgumentError] with a descriptive message listing missing tokens.
  void _validateSpecialTokens() {
    final List<String> required = <String>[
      _tokens.cls,
      _tokens.sep,
      _tokens.pad,
      _tokens.unk,
    ];

    final List<String> missing =
        required.where((String t) => !_vocab.containsKey(t)).toList();

    if (missing.isNotEmpty) {
      throw ArgumentError(
        'The following special tokens are missing from the vocabulary: '
        '${missing.join(', ')}. '
        'Ensure your vocab.txt was generated for the same model as your '
        'SpecialTokens configuration.',
      );
    }
  }
}
