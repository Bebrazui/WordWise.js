// src/utils/generation-algorithms.ts
import { Tensor } from '../lib/tensor';

/**
 * Применяет набор продвинутых эвристик и алгоритмов для выбора следующего слова.
 * Это основной "мозг" генерации, который делает текст осмысленным.
 * 
 * @param predictionLogits - Сырые логиты от модели [1, vocabSize].
 * @param indexToWord - Маппинг для преобразования индекса в слово.
 * @param temperature - Контролирует случайность.
 * @param generatedSequence - Уже сгенерированная последовательность для контекста.
 * @returns - Объект с выбранным словом и топ-5 предсказаниями.
 */
export function sampleNextWord(
  predictionLogits: Tensor,
  indexToWord: Map<number, string>,
  temperature: number = 0.8,
  generatedSequence: string[] = [],
): { chosenWord: string; topPredictions: { word: string; probability: number }[] } {
    if (predictionLogits.shape.length !== 2 || predictionLogits.shape[0] !== 1) {
        throw new Error(`Prediction tensor for word selection must be [1, vocabSize]. Got [${predictionLogits.shape}]`);
    }
    const vocabSize = predictionLogits.shape[1];
    const logits = predictionLogits.data.slice(); // Копируем, чтобы не изменять исходные логиты

    // --- Применяем температуру ---
    // Делаем это в начале, чтобы повлиять на распределение перед всеми эвристиками.
    for (let i = 0; i < vocabSize; i++) {
        logits[i] /= temperature;
    }

    // --- АЛГОРИТМ 1: Агрессивный штраф за повторение (Repetition Penalty) ---
    // Уменьшаем вероятность слов, которые недавно появлялись.
    const REPETITION_PENALTY = 1.9;
    const penaltyWindow = Math.min(generatedSequence.length, 15);
    const recentWords = new Set(generatedSequence.slice(-penaltyWindow));
    
    recentWords.forEach(word => {
        for (const [idx, w] of indexToWord.entries()) {
            if (w === word) {
                logits[idx] = logits[idx] < 0 ? logits[idx] * REPETITION_PENALTY : logits[idx] / REPETITION_PENALTY;
                break;
            }
        }
    });
    
    // --- АЛГОРИТМ 2: Блокировка N-граммов ---
    // Запрещаем генерацию n-граммов (последовательностей), которые уже были в тексте.
    const NGRAM_SIZE = 2; 
    if (generatedSequence.length >= NGRAM_SIZE -1) {
        const lastNMinus1Words = generatedSequence.slice(-(NGRAM_SIZE - 1));
        const currentNgramPrefix = lastNMinus1Words.join(' ');
        
        // Чтобы это было эффективнее, нужно было бы заранее построить индекс N-граммов,
        // но для простоты мы будем искать прямо в последовательности.
        for (let i = 0; i < vocabSize; i++) {
            const nextWord = indexToWord.get(i);
            if (!nextWord) continue;
            
            const potentialNgram = `${currentNgramPrefix} ${nextWord}`;
            if (generatedSequence.join(' ').includes(potentialNgram)) {
                logits[i] = -Infinity; // Блокируем этот токен, чтобы избежать повтора
            }
        }
    }

    // --- АЛГОРИТМ 3: Подавление специальных и коротких токенов ---
    for (let i = 0; i < vocabSize; i++) {
        const word = indexToWord.get(i);
        if (word === '<unk>' || word === 'вопрос' || word === 'ответ' || (word && word.length < 2 && word !== 'у' && word !== 'а' && word !== 'и')) {
            logits[i] = -Infinity; // Полностью блокируем эти токены
        }
    }

    // --- Softmax для получения вероятностей ---
    let maxLogit = -Infinity;
    for (const l of logits) {
        if (isFinite(l) && l > maxLogit) {
            maxLogit = l;
        }
    }
    // Если все логиты -Infinity, то мы в тупике.
    if (maxLogit === -Infinity) {
       const fallbackWord = indexToWord.get(0) || '<unk>';
       return { chosenWord: fallbackWord, topPredictions: [{word: fallbackWord, probability: 1.0}] };
    }

    const probabilities = new Float32Array(vocabSize);
    let sumExp = 0;
    for (let i = 0; i < vocabSize; i++) {
        if (isFinite(logits[i])) {
            const exp = Math.exp(logits[i] - maxLogit);
            probabilities[i] = exp;
            sumExp += exp;
        } else {
            probabilities[i] = 0;
        }
    }

    for (let i = 0; i < vocabSize; i++) {
        probabilities[i] /= sumExp;
    }

    // --- АЛГОРИТМ 4: Сэмплирование по ядру (Nucleus Sampling / Top-P) ---
    // Это более продвинутый метод, чем Top-K.
    const TOP_P = 0.92;
    const predictionsWithIndices = Array.from(probabilities)
        .map((probability, index) => ({
            word: indexToWord.get(index) || '<unk>',
            probability,
            index,
        }))
        .filter(p => p.probability > 0);

    predictionsWithIndices.sort((a, b) => b.probability - a.probability);

    const nucleus: typeof predictionsWithIndices = [];
    let cumulativeProbability = 0;
    for (const p of predictionsWithIndices) {
        nucleus.push(p);
        cumulativeProbability += p.probability;
        if (cumulativeProbability >= TOP_P) {
            break;
        }
    }

    if (nucleus.length === 0) {
        // Если ядро пустое (что маловероятно), берем лучший токен
        const fallback = predictionsWithIndices[0] || { index: 0, word: '<unk>', probability: 1.0 };
        nucleus.push(fallback);
    }
    
    // Пересчитываем вероятности внутри ядра
    const nucleusSum = nucleus.reduce((sum, p) => sum + p.probability, 0);
    const nucleusNormalized = nucleus.map(p => ({ ...p, probability: p.probability / nucleusSum }));

    // --- Финальный выбор слова (сэмплирование из ядра) ---
    let randomValue = Math.random();
    let chosenIndex = nucleusNormalized[0].index; // Fallback to the best one

    for (const pred of nucleusNormalized) {
        randomValue -= pred.probability;
        if (randomValue <= 0) {
            chosenIndex = pred.index;
            break;
        }
    }

    const chosenWord = indexToWord.get(chosenIndex) || '<unk>';
    const top5ForDisplay = predictionsWithIndices.slice(0, 5);

    return { chosenWord, topPredictions: top5ForDisplay };
}
