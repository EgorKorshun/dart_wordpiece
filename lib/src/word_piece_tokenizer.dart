import 'dart:io';
import 'dart:isolate';

import 'encoding_options.dart';
import 'special_tokens.dart';
import 'text_normalizer.dart';
import 'tokenizer_config.dart';
import 'tokenizer_output.dart';
import 'vocab_loader.dart';

// ---------------------------------------------------------------------------
// Internal record types
// ---------------------------------------------------------------------------

/// Result of the WordPiece algorithm: token strings paired with their
/// character-span offsets in the normalized input string.
typedef _WordPieceResult = ({
  List<String> tokens,
  List<(int, int)> offsets,
});

/// Full tokenization result including vocabulary IDs, offsets, and token
/// strings (needed for overflowing-tokens tracking).
typedef _TokenizeResult = ({
  List<int> ids,
  List<(int, int)> offsets,
  List<String> tokens,
});

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
/// print(output.offsetMapping); // [(0,0),(0,4),(5,7),(8,15),(0,0),…]
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
  /// print(out.offsetMapping); // [(0,0), (0,7), …, (0,0), (0,0), …]
  /// ```
  TokenizerOutput encode(String text) {
    final _TokenizeResult r = _tokenizeToResult(text);
    return _buildOutput(
      segmentA: r.ids,
      offsetsA: r.offsets,
      tokensA: r.tokens,
    );
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
    final _TokenizeResult rA = _tokenizeToResult(textA);
    final _TokenizeResult rB = _tokenizeToResult(textB);
    return _buildOutput(
      segmentA: rA.ids,
      offsetsA: rA.offsets,
      tokensA: rA.tokens,
      segmentB: rB.ids,
      offsetsB: rB.offsets,
      tokensB: rB.tokens,
    );
  }

  /// Encodes every string in [texts] and returns a list of [TokenizerOutput].
  ///
  /// With [PaddingStrategy.fixed] (default) every output has the same length
  /// ([TokenizerConfig.maxLength]), making it straightforward to stack into a
  /// batch tensor.
  ///
  /// With [PaddingStrategy.longest] every output is padded to the length of
  /// the longest sequence in [texts], which can save compute and memory.
  ///
  /// Example:
  /// ```dart
  /// final outputs = tokenizer.encodeAll(['hello', 'world']);
  /// ```
  List<TokenizerOutput> encodeAll(List<String> texts) {
    if (_config.paddingStrategy == PaddingStrategy.fixed) {
      return texts.map(encode).toList();
    }

    // PaddingStrategy.longest — two-pass.
    final List<_TokenizeResult> results =
        texts.map(_tokenizeToResult).toList();

    // Compute the real length each sequence would have after specials +
    // truncation (before padding).
    const int reserved = 2; // CLS + SEP
    int longestReal = 0;
    for (final _TokenizeResult r in results) {
      final int real =
          r.ids.length.clamp(0, _maxLength - reserved) + reserved;
      if (real > longestReal) longestReal = real;
    }

    final int padTarget = longestReal.clamp(2, _maxLength);
    return results
        .map(
          (_TokenizeResult r) => _buildOutput(
            segmentA: r.ids,
            offsetsA: r.offsets,
            tokensA: r.tokens,
            overridePadTarget: padTarget,
          ),
        )
        .toList();
  }

  /// Async variant of [encode].
  ///
  /// Uses [Future.value] — executes synchronously on the current isolate but
  /// returns a [Future] for `await` compatibility. For true CPU offloading use:
  /// ```dart
  /// final output = await Isolate.run(() => tokenizer.encode(text));
  /// ```
  Future<TokenizerOutput> encodeAsync(String text) =>
      Future.value(encode(text));

  /// Async variant of [encodePair].
  ///
  /// Uses [Future.value] — see [encodeAsync] for notes on isolate offloading.
  Future<TokenizerOutput> encodePairAsync(String textA, String textB) =>
      Future.value(encodePair(textA, textB));

  /// Async variant of [encodeAll].
  ///
  /// Offloads work to a separate [Isolate] because batch tokenization can be
  /// CPU-intensive. The vocabulary map is copied into the isolate by the Dart
  /// message-passing system.
  ///
  /// For very large vocabularies, constructing the tokenizer inside the
  /// isolate avoids the copy overhead:
  /// ```dart
  /// final outputs = await Isolate.run(() {
  ///   final t = WordPieceTokenizer(vocab: myVocab, config: myConfig);
  ///   return t.encodeAll(texts);
  /// });
  /// ```
  Future<List<TokenizerOutput>> encodeAllAsync(List<String> texts) =>
      Isolate.run(() => encodeAll(texts));

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

    final _WordPieceResult result = _wordpieceEncode(input);
    return <String>[_tokens.cls, ...result.tokens, _tokens.sep];
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

  /// Applies the WordPiece algorithm to [text] and returns tokens with their
  /// character-span offsets within [text].
  ///
  /// [text] must already be normalized (whitespace-collapsed) before calling
  /// this method. The character cursor advances one position per space between
  /// words, which matches the output of [TextNormalizer.normalize].
  ///
  /// The algorithm processes each whitespace-delimited word independently:
  ///
  /// 1. Try to find the longest prefix of [remaining] in the vocabulary.
  ///    - For the first piece of a word: look up [piece] directly.
  ///    - For continuation pieces: look up `##[piece]`.
  /// 2. If a match is found, emit the token with its char span and repeat on
  ///    the suffix.
  /// 3. If no match is found (not even a single character), emit
  ///    [SpecialTokens.unk] with a span covering the full unmatched word.
  _WordPieceResult _wordpieceEncode(String text) {
    final List<String> tokens = <String>[];
    final List<(int, int)> offsets = <(int, int)>[];

    final List<String> words = text.split(RegExp(r'\s+'));
    int charCursor = 0;

    for (final String word in words) {
      if (word.isEmpty) {
        charCursor++;
        continue;
      }

      final int wordStart = charCursor;
      String remaining = word;
      bool isFirstPiece = true;
      int consumed = 0;

      while (remaining.isNotEmpty) {
        bool matched = false;

        // Greedy longest-match-first search from remaining.length down to 1.
        for (int end = remaining.length; end > 0; end--) {
          final String candidate = remaining.substring(0, end);
          final String key =
              isFirstPiece ? candidate : '${_tokens.subwordPrefix}$candidate';

          if (_vocab.containsKey(key)) {
            tokens.add(key);
            offsets.add((wordStart + consumed, wordStart + consumed + end));
            remaining = remaining.substring(end);
            consumed += end;
            isFirstPiece = false;
            matched = true;
            break;
          }
        }

        if (!matched) {
          // No vocabulary match for any prefix → replace whole word with UNK.
          tokens.add(_tokens.unk);
          offsets.add((wordStart, wordStart + word.length));
          break;
        }
      }

      charCursor += word.length + 1; // word chars + one inter-word space
    }

    return (tokens: tokens, offsets: offsets);
  }

  // ---------------------------------------------------------------------------
  // Encoding helpers
  // ---------------------------------------------------------------------------

  /// Normalizes [text], runs WordPiece, and returns IDs, offsets, and token
  /// strings in one shot (avoids double-encoding).
  _TokenizeResult _tokenizeToResult(String text) {
    final String input =
        _config.normalizeText ? _normalizer.normalize(text) : text.toLowerCase();
    final _WordPieceResult wp = _wordpieceEncode(input);
    final int unkId = _vocab[_tokens.unk]!;
    return (
      ids: wp.tokens.map((String t) => _vocab[t] ?? unkId).toList(),
      offsets: wp.offsets,
      tokens: wp.tokens,
    );
  }

  /// Assembles the final [TokenizerOutput] from pre-tokenized segment data.
  ///
  /// Single-segment: `[CLS] segA [SEP] [PAD]…`
  /// Two-segment:    `[CLS] segA [SEP] segB [SEP] [PAD]…`
  ///
  /// Truncates [segmentB] first when the total exceeds the pad target.
  /// Uses [TokenizerConfig.truncationSide] to decide which side to drop.
  TokenizerOutput _buildOutput({
    required List<int> segmentA,
    required List<(int, int)> offsetsA,
    required List<String> tokensA,
    List<int>? segmentB,
    List<(int, int)>? offsetsB,
    List<String>? tokensB,
    int? overridePadTarget,
  }) {
    final int clsId = _vocab[_tokens.cls]!;
    final int sepId = _vocab[_tokens.sep]!;
    final int padId = _vocab[_tokens.pad]!;

    final int padTarget = overridePadTarget ?? _maxLength;

    // Reserve slots: 1 CLS + 1 SEP for segment A (+ 1 SEP for segment B).
    final int reservedSlots = segmentB == null ? 2 : 3;
    final int available = padTarget - reservedSlots;

    List<int> a = segmentA;
    List<(int, int)> aOff = offsetsA;
    List<String> aTok = tokensA;

    List<int> b = segmentB ?? <int>[];
    List<(int, int)> bOff = offsetsB ?? <(int, int)>[];
    List<String> bTok = tokensB ?? <String>[];

    // Collect overflowing tokens before truncation.
    final List<String> overflowing = <String>[];

    if (a.length + b.length > available) {
      if (_config.truncationSide == TruncationSide.right) {
        // Truncate from the end: segment B first, then segment A.
        final int bAllowed = (available - a.length).clamp(0, b.length);
        overflowing.addAll(bTok.sublist(bAllowed));
        b = b.sublist(0, bAllowed);
        bOff = bOff.sublist(0, bAllowed);
        bTok = bTok.sublist(0, bAllowed);

        final int aAllowed = (available - b.length).clamp(0, a.length);
        overflowing.addAll(aTok.sublist(aAllowed));
        a = a.sublist(0, aAllowed);
        aOff = aOff.sublist(0, aAllowed);
        aTok = aTok.sublist(0, aAllowed);
      } else {
        // TruncationSide.left — keep the tail of each segment.
        final int bAllowed = (available - a.length).clamp(0, b.length);
        overflowing.addAll(bTok.sublist(0, b.length - bAllowed));
        b = b.sublist(b.length - bAllowed);
        bOff = bOff.sublist(bOff.length - bAllowed);
        bTok = bTok.sublist(bTok.length - bAllowed);

        final int aAllowed = (available - b.length).clamp(0, a.length);
        overflowing.addAll(aTok.sublist(0, a.length - aAllowed));
        a = a.sublist(a.length - aAllowed);
        aOff = aOff.sublist(aOff.length - aAllowed);
        aTok = aTok.sublist(aTok.length - aAllowed);
      }
    }

    // Build flat token ID list.
    final List<int> ids = <int>[clsId, ...a, sepId, ...b];
    if (segmentB != null) ids.add(sepId);

    // Build token_type_ids: 0 for CLS + segA + SEP, 1 for segB + SEP.
    final List<int> typeIds = <int>[
      ...List<int>.filled(a.length + 2, 0), // CLS + segA + SEP
      ...List<int>.filled(b.length + (segmentB != null ? 1 : 0), 1),
    ];

    // Attention mask: 1 for real tokens.
    final int realCount = ids.length;
    final List<int> mask = List<int>.filled(realCount, 1, growable: true);

    // specialTokensMask: 1 for CLS/SEP, 0 for content.
    final List<int> specialMask = <int>[
      1, // CLS
      ...List<int>.filled(a.length, 0),
      1, // SEP after A
      ...List<int>.filled(b.length, 0),
      if (segmentB != null) 1, // SEP after B
    ];

    // offsetMapping: (0,0) sentinels for special tokens.
    final List<(int, int)> offsetMap = <(int, int)>[
      (0, 0), // CLS
      ...aOff,
      (0, 0), // SEP after A
      ...bOff,
      if (segmentB != null) (0, 0), // SEP after B
    ];

    // Pad all lists to padTarget.
    while (ids.length < padTarget) {
      ids.add(padId);
      mask.add(0);
      typeIds.add(0);
      specialMask.add(1); // PAD is a special token
      offsetMap.add((0, 0));
    }

    return TokenizerOutput(
      inputIds: ids,
      attentionMask: mask,
      tokenTypeIds: typeIds,
      offsetMapping: offsetMap,
      specialTokensMask: specialMask,
      overflowingTokens: overflowing,
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
