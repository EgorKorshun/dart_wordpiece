// Tests use explicit default values to make expected behaviour obvious.
// ignore_for_file: avoid_redundant_argument_values

import 'package:dart_wordpiece/dart_wordpiece.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Minimal toy vocabulary used across all tests.
//
// IDs follow the bert-base-uncased convention for special tokens so that
// tests read naturally:
//   0  → [PAD]
//   1  → [UNK]
//   2  → [CLS]   (bert-base uses 101, but small IDs simplify assertions)
//   3  → [SEP]
//   4  → flutter
//   5  → is
//   6  → fast
//   7  → dart
//   8  → ##ing   (continuation piece)
//   9  → play
//  10  → un
//  11  → ##aff
//  12  → ##able
// ---------------------------------------------------------------------------
Map<String, int> _buildVocab() => {
      '[PAD]': 0,
      '[UNK]': 1,
      '[CLS]': 2,
      '[SEP]': 3,
      'flutter': 4,
      'is': 5,
      'fast': 6,
      'dart': 7,
      '##ing': 8,
      'play': 9,
      'un': 10,
      '##aff': 11,
      '##able': 12,
    };

/// Creates a tokenizer backed by the toy vocabulary.
WordPieceTokenizer _tokenizer({
  int maxLength = 16,
  Set<String> stopwords = const <String>{},
  bool normalizeText = true,
}) =>
    WordPieceTokenizer(
      vocab: _buildVocab(),
      config: TokenizerConfig(
        maxLength: maxLength,
        stopwords: stopwords,
        normalizeText: normalizeText,
      ),
    );

void main() {
  // =========================================================================
  // SpecialTokens
  // =========================================================================
  group('SpecialTokens', () {
    test('bert() constructor sets expected values', () {
      const tokens = SpecialTokens.bert();
      expect(tokens.cls, '[CLS]');
      expect(tokens.sep, '[SEP]');
      expect(tokens.pad, '[PAD]');
      expect(tokens.unk, '[UNK]');
      expect(tokens.subwordPrefix, '##');
    });

    test('custom constructor preserves all fields', () {
      const tokens = SpecialTokens(
        cls: '<s>',
        sep: '</s>',
        pad: '<pad>',
        unk: '<unk>',
        subwordPrefix: 'Ġ',
      );
      expect(tokens.cls, '<s>');
      expect(tokens.subwordPrefix, 'Ġ');
    });

    test('equality holds for identical values', () {
      const a = SpecialTokens.bert();
      const b = SpecialTokens.bert();
      expect(a, equals(b));
    });

    test('inequality when any field differs', () {
      const a = SpecialTokens.bert();
      const b = SpecialTokens(
        cls: '[CLS]',
        sep: '[SEP]',
        pad: '[PAD]',
        unk: '[UNK]',
        subwordPrefix: 'Ġ', // different prefix
      );
      expect(a, isNot(equals(b)));
    });
  });

  // =========================================================================
  // TokenizerConfig
  // =========================================================================
  group('TokenizerConfig', () {
    test('default values match BERT baseline', () {
      const config = TokenizerConfig();
      expect(config.maxLength, 64);
      expect(config.normalizeText, isTrue);
      expect(config.stopwords, isEmpty);
      expect(config.specialTokens, const SpecialTokens.bert());
    });

    test('custom values are preserved', () {
      const config = TokenizerConfig(
        maxLength: 128,
        stopwords: {'the', 'a'},
        normalizeText: false,
      );
      expect(config.maxLength, 128);
      expect(config.stopwords, containsAll(['the', 'a']));
      expect(config.normalizeText, isFalse);
    });
  });

  // =========================================================================
  // TextNormalizer
  // =========================================================================
  group('TextNormalizer', () {
    const normalizer = TextNormalizer(
      stopWords: {'what', 'is', 'the', 'a', 'how'},
    );

    test('returns empty string unchanged', () {
      expect(normalizer.normalize(''), '');
    });

    test('lowercases input', () {
      expect(normalizer.normalize('Flutter IS FAST'), 'flutter fast');
    });

    test('removes punctuation', () {
      expect(normalizer.normalize('Hello, world!'), 'hello world');
    });

    test('strips stopwords', () {
      expect(normalizer.normalize('What is the answer'), 'answer');
    });

    test('collapses multiple spaces', () {
      expect(normalizer.normalize('a   b   c'), 'b c');
    });

    test('trims leading and trailing whitespace', () {
      expect(normalizer.normalize('  flutter  '), 'flutter');
    });

    test('falls back to lowercased text when all words are stopwords', () {
      expect(normalizer.normalize('What is a'), 'what is a');
    });

    test('contentWords splits normalized output', () {
      // 'how' is a stopword → removed; 'does' is not → kept.
      expect(
        normalizer.contentWords('How does Dart handle async?'),
        ['does', 'dart', 'handle', 'async'],
      );
    });

    test('isStopword returns true for known stopword', () {
      expect(normalizer.isStopWord('is'), isTrue);
    });

    test('isStopword returns false for non-stopword', () {
      expect(normalizer.isStopWord('flutter'), isFalse);
    });
  });

  // =========================================================================
  // VocabLoader
  // =========================================================================
  group('VocabLoader', () {
    test('fromString parses one-token-per-line format', () {
      const content = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello';
      final vocab = VocabLoader.fromString(content);
      expect(vocab['[PAD]'], 0);
      expect(vocab['[UNK]'], 1);
      expect(vocab['[CLS]'], 2);
      expect(vocab['hello'], 4);
    });

    test('fromString skips blank lines and preserves IDs', () {
      const content = 'a\n\nb\n\nc';
      final vocab = VocabLoader.fromString(content);
      // Blank lines are counted in the index, so IDs jump.
      expect(vocab['a'], 0);
      expect(vocab['b'], 2);
      expect(vocab['c'], 4);
    });

    test('fromMap returns the same map', () {
      final map = {'[PAD]': 0, 'hello': 1};
      final vocab = VocabLoader.fromMap(map);
      expect(vocab, same(map));
    });

    test('vocabSize returns number of tokens', () {
      final vocab = VocabLoader.fromString('[PAD]\n[UNK]\nhello');
      expect(VocabLoader.vocabSize(vocab), 3);
    });

    test('contains returns true for known token', () {
      final vocab = VocabLoader.fromMap({'[PAD]': 0});
      expect(VocabLoader.contains(vocab, '[PAD]'), isTrue);
    });

    test('contains returns false for unknown token', () {
      final vocab = VocabLoader.fromMap({'[PAD]': 0});
      expect(VocabLoader.contains(vocab, 'xyz'), isFalse);
    });
  });

  // =========================================================================
  // TokenizerOutput
  // =========================================================================
  group('TokenizerOutput', () {
    const out = TokenizerOutput(
      inputIds: [2, 4, 5, 3, 0, 0],
      attentionMask: [1, 1, 1, 1, 0, 0],
      tokenTypeIds: [0, 0, 0, 0, 0, 0],
    );

    test('length equals list size', () => expect(out.length, 6));
    test('realLength counts mask=1 positions', () => expect(out.realLength, 4));

    test('inputIdsInt64 has correct values', () {
      final i64 = out.inputIdsInt64;
      expect(i64[0], 2);
      expect(i64.length, 6);
    });

    test('equality holds for identical data', () {
      const other = TokenizerOutput(
        inputIds: [2, 4, 5, 3, 0, 0],
        attentionMask: [1, 1, 1, 1, 0, 0],
        tokenTypeIds: [0, 0, 0, 0, 0, 0],
      );
      expect(out, equals(other));
    });

    test('inequality when inputIds differ', () {
      const other = TokenizerOutput(
        inputIds: [2, 7, 5, 3, 0, 0], // 7 != 4
        attentionMask: [1, 1, 1, 1, 0, 0],
        tokenTypeIds: [0, 0, 0, 0, 0, 0],
      );
      expect(out, isNot(equals(other)));
    });
  });

  // =========================================================================
  // WordPieceTokenizer — construction
  // =========================================================================
  group('WordPieceTokenizer construction', () {
    test('vocabSize returns number of tokens in vocabulary', () {
      expect(_tokenizer().vocabSize, _buildVocab().length);
    });

    test('throws ArgumentError when special token is missing', () {
      expect(
        () => WordPieceTokenizer(
          vocab: {'only_one_token': 0}, // missing [CLS], [SEP], etc.
        ),
        throwsArgumentError,
      );
    });

    test('fromString factory creates tokenizer from vocab text', () {
      final raw = _buildVocab().entries
          .toList()
        ..sort((a, b) => a.value.compareTo(b.value));
      final content = raw.map((e) => e.key).join('\n');
      final t = WordPieceTokenizer.fromString(content);
      expect(t.vocabSize, _buildVocab().length);
    });
  });

  // =========================================================================
  // WordPieceTokenizer — tokenize()
  // =========================================================================
  group('WordPieceTokenizer.tokenize', () {
    late WordPieceTokenizer t;
    setUp(() => t = _tokenizer());

    test('wraps output with [CLS] and [SEP]', () {
      final tokens = t.tokenize('flutter');
      expect(tokens.first, '[CLS]');
      expect(tokens.last, '[SEP]');
    });

    test('known word maps to its token', () {
      final tokens = t.tokenize('flutter');
      expect(tokens, containsAllInOrder(['[CLS]', 'flutter', '[SEP]']));
    });

    test('unknown word becomes [UNK]', () {
      final tokens = t.tokenize('xyzzy');
      expect(tokens, contains('[UNK]'));
    });

    test('word split into subword pieces with ## prefix', () {
      // "playing" → "play" + "##ing"
      final tokens = t.tokenize('playing');
      expect(tokens, containsAllInOrder(['[CLS]', 'play', '##ing', '[SEP]']));
    });

    test('multi-word input tokenizes each word independently', () {
      final tokens = t.tokenize('flutter is fast');
      expect(
        tokens,
        containsAllInOrder(['[CLS]', 'flutter', 'is', 'fast', '[SEP]']),
      );
    });
  });

  // =========================================================================
  // WordPieceTokenizer — encode()
  // =========================================================================
  group('WordPieceTokenizer.encode', () {
    late WordPieceTokenizer t;
    setUp(() => t = _tokenizer(maxLength: 8));

    test('output length equals maxLength', () {
      expect(t.encode('flutter').length, 8);
    });

    test('first inputId is [CLS] token ID', () {
      expect(t.encode('flutter').inputIds.first, 2); // [CLS] = 2
    });

    test('last real token ID is [SEP]', () {
      final out = t.encode('flutter');
      final lastReal = out.inputIds[out.realLength - 1];
      expect(lastReal, 3); // [SEP] = 3
    });

    test('pad positions have inputId = [PAD] token ID', () {
      final out = t.encode('flutter');
      final padSlots = out.inputIds.sublist(out.realLength);
      expect(padSlots.every((id) => id == 0), isTrue); // [PAD] = 0
    });

    test('attentionMask is 1 for real tokens and 0 for pads', () {
      final out = t.encode('flutter');
      expect(out.attentionMask.sublist(0, out.realLength), everyElement(1));
      expect(out.attentionMask.sublist(out.realLength), everyElement(0));
    });

    test('all tokenTypeIds are 0 for single-segment input', () {
      final out = t.encode('flutter is fast');
      expect(out.tokenTypeIds, everyElement(0));
    });

    test('long sequence is truncated to maxLength', () {
      // 6 tokens "flutter is fast dart play ##ing" + CLS + SEP = 8 → fits
      // Add one more to force truncation.
      final tShort = _tokenizer(maxLength: 6);
      final out = tShort.encode('flutter is fast dart');
      expect(out.length, 6);
      expect(out.inputIds.last, anyOf(0, 3)); // last real = SEP or pad
    });

    test('realLength increases with longer input', () {
      final short = t.encode('flutter');
      final long = t.encode('flutter is fast');
      expect(long.realLength, greaterThan(short.realLength));
    });

    test('tokenToId returns correct ID', () {
      expect(t.tokenToId('flutter'), 4);
    });

    test('tokenToId returns null for unknown token', () {
      expect(t.tokenToId('xyz'), isNull);
    });
  });

  // =========================================================================
  // WordPieceTokenizer — encodePair()
  // =========================================================================
  group('WordPieceTokenizer.encodePair', () {
    late WordPieceTokenizer t;
    setUp(() => t = _tokenizer(maxLength: 16));

    test('output length equals maxLength', () {
      expect(t.encodePair('flutter', 'dart').length, 16);
    });

    test('segment A tokens have tokenTypeId = 0', () {
      // Encode "flutter" + "dart":
      // [CLS](0) flutter(0) [SEP](0) dart(1) [SEP](1) [PAD](0)…
      final out = t.encodePair('flutter', 'dart');
      // Positions 0-2 (CLS, flutter, SEP) should be type 0.
      expect(out.tokenTypeIds[0], 0);
      expect(out.tokenTypeIds[1], 0);
      expect(out.tokenTypeIds[2], 0);
    });

    test('segment B tokens have tokenTypeId = 1', () {
      final out = t.encodePair('flutter', 'dart');
      // Position 3 = "dart" → type 1; position 4 = SEP for B → type 1.
      expect(out.tokenTypeIds[3], 1);
      expect(out.tokenTypeIds[4], 1);
    });

    test('padding positions have tokenTypeId = 0', () {
      final out = t.encodePair('flutter', 'dart');
      final padStart = out.realLength;
      final padTypes = out.tokenTypeIds.sublist(padStart);
      expect(padTypes, everyElement(0));
    });

    test('two SEP tokens appear in inputIds', () {
      final out = t.encodePair('flutter', 'dart');
      expect(
        out.inputIds.sublist(0, out.realLength).where((id) => id == 3).length,
        2, // one SEP after A, one after B
      );
    });
  });

  // =========================================================================
  // WordPieceTokenizer — encodeAll()
  // =========================================================================
  group('WordPieceTokenizer.encodeAll', () {
    test('returns list of same length as input', () {
      final t = _tokenizer();
      final results = t.encodeAll(['flutter', 'dart', 'fast']);
      expect(results.length, 3);
    });

    test('each output has the same length', () {
      final t = _tokenizer(maxLength: 8);
      final results = t.encodeAll(['flutter', 'dart is fast']);
      expect(results.every((o) => o.length == 8), isTrue);
    });
  });

  // =========================================================================
  // WordPieceTokenizer — stopword filtering
  // =========================================================================
  group('WordPieceTokenizer stopword filtering', () {
    test('stopwords are removed before tokenization', () {
      final t = _tokenizer(stopwords: {'is'});
      // "flutter is fast" → normalizes to "flutter fast"
      final tokens = t.tokenize('flutter is fast');
      expect(tokens, isNot(contains('is')));
      expect(tokens, containsAllInOrder(['[CLS]', 'flutter', 'fast', '[SEP]']));
    });

    test('stopword removal does not affect tokens when disabled', () {
      final t = _tokenizer(stopwords: {'is'}, normalizeText: false);
      // normalizeText=false → stopwords not applied
      final tokens = t.tokenize('flutter is fast');
      expect(tokens, contains('is'));
    });
  });

  // =========================================================================
  // v1.2.0 — offsetMapping
  // =========================================================================
  group('TokenizerOutput.offsetMapping', () {
    test('CLS and SEP have sentinel (0,0)', () {
      final t = _tokenizer(maxLength: 8);
      final out = t.encode('flutter');
      // CLS at index 0, SEP at index 2
      expect(out.offsetMapping![0], (0, 0));
      expect(out.offsetMapping![2], (0, 0));
    });

    test('PAD positions have sentinel (0,0)', () {
      final t = _tokenizer(maxLength: 8);
      final out = t.encode('flutter'); // realLength=3, rest is padding
      for (int i = 3; i < 8; i++) {
        expect(out.offsetMapping![i], (0, 0),
            reason: 'index $i should be a PAD sentinel');
      }
    });

    test('whole-word token gets span covering the full word', () {
      final t = _tokenizer(maxLength: 8);
      // "flutter" (7 chars) → single token at offset (0,7)
      final out = t.encode('flutter');
      expect(out.offsetMapping![1], (0, 7));
    });

    test('subword pieces get correct spans within word', () {
      final t = _tokenizer(maxLength: 8);
      // "playing" → play(0,4) + ##ing(4,7)
      // Vocab has: play=9, ##ing=8 — "playing" is not in vocab so it splits
      final out = t.encode('playing');
      expect(out.offsetMapping![1], (0, 4)); // play
      expect(out.offsetMapping![2], (4, 7)); // ##ing
    });

    test('multi-word sequence: second word starts after space', () {
      final t = _tokenizer(maxLength: 8);
      // "flutter fast" → flutter at (0,7), fast at (8,12)
      final out = t.encode('flutter fast');
      expect(out.offsetMapping![1], (0, 7)); // flutter
      expect(out.offsetMapping![2], (8, 12)); // fast
    });

    test('offsetMapping length equals inputIds length', () {
      final t = _tokenizer(maxLength: 10);
      final out = t.encode('flutter is fast');
      expect(out.offsetMapping!.length, out.inputIds.length);
    });

    test('encodePair: segA and segB use independent spans', () {
      final t = _tokenizer(maxLength: 10);
      // A: "flutter" (7 chars) → (0,7)
      // B: "fast" (4 chars) → (0,4)
      final out = t.encodePair('flutter', 'fast');
      expect(out.offsetMapping![1], (0, 7)); // flutter in seg A
      expect(out.offsetMapping![3], (0, 4)); // fast in seg B (spans reset)
    });
  });

  // =========================================================================
  // v1.2.0 — specialTokensMask
  // =========================================================================
  group('TokenizerOutput.specialTokensMask', () {
    test('CLS, SEP, PAD are 1; content tokens are 0', () {
      final t = _tokenizer(maxLength: 8);
      final out = t.encode('flutter'); // [CLS flutter SEP PAD PAD PAD PAD PAD]
      expect(out.specialTokensMask, [1, 0, 1, 1, 1, 1, 1, 1]);
    });

    test('length matches inputIds length', () {
      final t = _tokenizer(maxLength: 12);
      final out = t.encode('flutter is fast');
      expect(out.specialTokensMask!.length, out.inputIds.length);
    });

    test('pair encoding: both SEP tokens are masked as 1', () {
      final t = _tokenizer(maxLength: 8);
      // [CLS flutter SEP fast SEP PAD PAD PAD]
      //   1    0      1    0    1   1   1   1
      final out = t.encodePair('flutter', 'fast');
      expect(out.specialTokensMask, [1, 0, 1, 0, 1, 1, 1, 1]);
    });
  });

  // =========================================================================
  // v1.2.0 — overflowingTokens
  // =========================================================================
  group('TokenizerOutput.overflowingTokens', () {
    test('empty list when sequence fits within maxLength', () {
      final t = _tokenizer(maxLength: 16);
      final out = t.encode('flutter is fast');
      expect(out.overflowingTokens, isEmpty);
    });

    test('right truncation captures dropped tail tokens', () {
      // "flutter is fast dart" = 4 tokens; maxLength=5 → available=3
      // Kept: flutter, is, fast  — dropped: dart
      final t = _tokenizer(maxLength: 5);
      final out = t.encode('flutter is fast dart');
      expect(out.overflowingTokens, ['dart']);
    });

    test('left truncation captures dropped head tokens', () {
      final t = WordPieceTokenizer(
        vocab: _buildVocab(),
        config: const TokenizerConfig(
          maxLength: 5,
          truncationSide: TruncationSide.left,
        ),
      );
      // available=3; last 3 tokens of [flutter, is, fast, dart] = [is, fast, dart]
      // dropped from head: [flutter]
      final out = t.encode('flutter is fast dart');
      expect(out.overflowingTokens, ['flutter']);
    });
  });

  // =========================================================================
  // v1.2.0 — PaddingStrategy.longest
  // =========================================================================
  group('PaddingStrategy.longest', () {
    test('batch pads to longest sequence length, not maxLength', () {
      final t = WordPieceTokenizer(
        vocab: _buildVocab(),
        config: const TokenizerConfig(
          maxLength: 16,
          paddingStrategy: PaddingStrategy.longest,
        ),
      );
      // 'flutter' → 1 token → realLength 3 (CLS + token + SEP)
      // 'fast'    → 1 token → realLength 3
      final outputs = t.encodeAll(['flutter', 'fast']);
      expect(outputs[0].length, 3);
      expect(outputs[1].length, 3);
    });

    test('when sequences differ, all pad to longest', () {
      final t = WordPieceTokenizer(
        vocab: _buildVocab(),
        config: const TokenizerConfig(
          maxLength: 16,
          paddingStrategy: PaddingStrategy.longest,
        ),
      );
      // 'flutter' → real 3; 'flutter is fast' → real 5
      final outputs = t.encodeAll(['flutter', 'flutter is fast']);
      expect(outputs[0].length, 5);
      expect(outputs[1].length, 5);
    });

    test('encode() always uses fixed padding regardless of config', () {
      final t = WordPieceTokenizer(
        vocab: _buildVocab(),
        config: const TokenizerConfig(
          maxLength: 16,
          paddingStrategy: PaddingStrategy.longest,
        ),
      );
      final out = t.encode('flutter');
      // Single encode always pads to maxLength
      expect(out.length, 16);
    });
  });

  // =========================================================================
  // v1.2.0 — TruncationSide.left
  // =========================================================================
  group('TruncationSide.left', () {
    test('left truncation keeps the last N tokens of a long segment', () {
      final t = WordPieceTokenizer(
        vocab: _buildVocab(),
        config: const TokenizerConfig(
          maxLength: 5,
          truncationSide: TruncationSide.left,
        ),
      );
      // available=3; [flutter, is, fast, dart] → keep last 3: [is, fast, dart]
      final out = t.encode('flutter is fast dart');
      expect(out.inputIds[1], 5); // 'is'
      expect(out.inputIds[2], 6); // 'fast'
      expect(out.inputIds[3], 7); // 'dart'
    });

    test('right truncation (default) keeps the first N tokens', () {
      final t = _tokenizer(maxLength: 5);
      // available=3; [flutter, is, fast, dart] → keep first 3: [flutter, is, fast]
      final out = t.encode('flutter is fast dart');
      expect(out.inputIds[1], 4); // 'flutter'
      expect(out.inputIds[2], 5); // 'is'
      expect(out.inputIds[3], 6); // 'fast'
    });
  });

  // =========================================================================
  // v1.2.0 — async variants
  // =========================================================================
  group('async variants', () {
    test('encodeAsync returns same result as encode', () async {
      final t = _tokenizer(maxLength: 8);
      final sync = t.encode('flutter is fast');
      final async_ = await t.encodeAsync('flutter is fast');
      expect(async_, sync);
    });

    test('encodePairAsync returns same result as encodePair', () async {
      final t = _tokenizer(maxLength: 10);
      final sync = t.encodePair('flutter', 'fast');
      final async_ = await t.encodePairAsync('flutter', 'fast');
      expect(async_, sync);
    });

    test('encodeAllAsync returns same result as encodeAll', () async {
      final t = _tokenizer(maxLength: 8);
      final texts = ['flutter', 'dart is fast'];
      final sync = t.encodeAll(texts);
      final async_ = await t.encodeAllAsync(texts);
      expect(async_.length, sync.length);
      for (int i = 0; i < sync.length; i++) {
        expect(async_[i], sync[i]);
      }
    });
  });
}
