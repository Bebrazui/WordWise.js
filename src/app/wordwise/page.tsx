// src/app/wordwise/page.tsx
"use client";

import { useState, useRef, useEffect } from 'react';
import { Tensor } from '../../lib/tensor';
import { crossEntropyLossWithSoftmaxGrad, softmax } from '../../lib/layers';
import { SGD } from '../../lib/optimizer';
import { WordWiseModel } from '../../lib/model';
import { buildVocabulary, wordsToInputTensors, wordsToTargetTensors, getWordFromPrediction, createBatches } from '../../utils/tokenizer';
import Link from 'next/link';

export default function WordwisePage() {
  const [output, setOutput] = useState<string>('');
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [status, setStatus] = useState<string>('Готов к инициализации.');
  const modelRef = useRef<WordWiseModel | null>(null);
  const optimizerRef = useRef<SGD | null>(null);
  const vocabDataRef = useRef<{ vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } | null>(null);

  // Параметры модели (можно экспериментировать с ними)
  const embeddingDim = 32; // Размерность вектора эмбеддинга для каждого слова
  const hiddenSize = 64;   // Размерность скрытого состояния LSTM
  const batchSize = 4;     // Количество примеров, обрабатываемых за один шаг обучения
  const learningRate = 0.01; // Скорость обучения
  const numEpochs = 500;   // Количество полных проходов по данным

  // Пример текстового корпуса для обучения
  const textCorpus = "the cat sat on the mat. the dog ran fast. cat and dog are pets and friends. a cat and a dog play together. and the cat loves the dog.";

  /**
   * Инициализирует WordWise.js: строит словарь и создает модель.
   * @param textData Текстовый корпус для построения словаря.
   */
  const initializeWordWise = (textData: string) => {
    try {
      setStatus('Инициализация WordWise.js...');
      const { vocab, wordToIndex, indexToWord, vocabSize } = buildVocabulary(textData);
      vocabDataRef.current = { vocab, wordToIndex, indexToWord, vocabSize };

      modelRef.current = new WordWiseModel(vocabSize, embeddingDim, hiddenSize);
      optimizerRef.current = new SGD(learningRate);

      setLossHistory([]);
      setOutput('WordWise.js инициализирован. Словарь создан.');
      console.log('Словарь:', vocab);
      setStatus('Готов к обучению.');
    } catch (error) {
      console.error("Ошибка инициализации:", error);
      setStatus(`Ошибка инициализации: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  /**
   * Запускает процесс обучения модели WordWise.js.
   */
  const trainWordWise = async () => {
    // Проверяем, инициализирована ли модель
    if (!modelRef.current || !optimizerRef.current || !vocabDataRef.current) {
      initializeWordWise(textCorpus);
      if (!modelRef.current || !optimizerRef.current || !vocabDataRef.current) {
          setStatus('Инициализация не удалась, обучение невозможно.');
          return;
      }
    }

    const model = modelRef.current;
    const optimizer = optimizerRef.current;
    const { wordToIndex, vocabSize } = vocabDataRef.current;

    const words = textCorpus.toLowerCase().match(/\b\w+\b/g) || [];
    if (words.length < 2) {
        setStatus('Недостаточно слов для обучения в корпусе.');
        return;
    }

    // Подготовка данных для обучения: вход - текущее слово, цель - следующее слово
    const inputTensors = wordsToInputTensors(words.slice(0, -1), wordToIndex);
    const targetTensors = wordsToTargetTensors(words.slice(1), wordToIndex, vocabSize);

    // Создаем батчи из подготовленных тензоров
    const batches = createBatches(inputTensors, targetTensors, batchSize);
    console.log(`Корпус разбит на ${batches.length} батчей.`);
    
    const newLossHistory: number[] = [];
    setLossHistory(newLossHistory); // Сброс истории потерь
    setStatus('Начинается обучение WordWise.js...');
    setOutput('');

    for (let epoch = 0; epoch < numEpochs; epoch++) {
      let epochLoss = 0;

      // Инициализируем скрытые состояния для начала каждой эпохи (для новой последовательности)
      let h = model.initializeStates(batchSize).h0;
      let c = model.initializeStates(batchSize).c0;

      for (const batch of batches) {
        const inputs = batch.inputs; // [batchSize, 1] - тензор индексов слов
        const targets = batch.targets; // [batchSize, vocabSize] - тензор one-hot целевых слов

        // 1. Прямой проход для одного шага LSTM с батчем
        const { outputLogits: predictionLogits, h: nextH, c: nextC } = model.forwardStep(inputs, h, c);

        // Обновляем скрытые состояния для следующего шага
        h = nextH;
        c = nextC;

        // 2. Вычисление потерь (Cross-Entropy Loss со Softmax градиентом)
        const lossTensor = crossEntropyLossWithSoftmaxGrad(predictionLogits, targets);
        epochLoss += lossTensor.data[0]; // Аккумулируем потерю

        // 3. Обратный проход: вычисление градиентов по всему графу
        lossTensor.backward();

        // 4. Обновление весов модели с помощью оптимизатора
        optimizer.step(model.getParameters());
      }

      const avgEpochLoss = epochLoss / batches.length;
      newLossHistory.push(avgEpochLoss);
      
      setLossHistory([...newLossHistory]);

      // Логирование прогресса
      if (epoch % 50 === 0 || epoch === numEpochs - 1) {
        console.log(`Эпоха ${epoch + 1}/${numEpochs}, Средняя потеря: ${avgEpochLoss.toFixed(6)}`);
        setStatus(`Обучение: Эпоха ${epoch + 1}/${numEpochs}, Потеря: ${avgEpochLoss.toFixed(6)}`);
      }
      // Небольшая задержка, чтобы UI не зависал при интенсивном обучении
      if (epoch % 20 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    setStatus('Обучение завершено.');
    setOutput('Обучение WordWise.js завершено. Теперь можно генерировать текст!');
  };

  /**
   * Генерирует текст, используя обученную модель.
   * @param startWord Начальное слово для генерации.
   * @param numWords Количество слов для генерации.
   */
  const generateText = (startWord: string, numWords: number) => {
    if (!modelRef.current || !vocabDataRef.current) {
      setOutput('WordWise.js не обучен. Сначала инициализируйте и обучите его.');
      setStatus('Генерация невозможна: модель не обучена.');
      return;
    }

    const model = modelRef.current;
    const { wordToIndex, indexToWord, vocabSize } = vocabDataRef.current;

    let currentWord = startWord.toLowerCase();
    // Проверяем, есть ли начальное слово в словаре
    if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>'; // Если нет, используем токен для неизвестных слов
        setOutput(`Начальное слово "${startWord}" не найдено в словаре. Используем "<unk>".`);
    }

    let generatedSequence = [currentWord];
    // Инициализируем скрытые состояния для генерации (батч=1)
    let h = model.initializeStates(1).h0;
    let c = model.initializeStates(1).c0;

    setStatus(`Генерация текста, начало: "${currentWord}"...`);

    for (let i = 0; i < numWords; i++) {
      // Преобразуем текущее слово в тензор индекса
      const inputTensor = new Tensor([wordToIndex.get(currentWord) || wordToIndex.get('<unk>')!], [1]); // [1,1]

      // Прямой проход: получаем логиты и новые состояния
      const { outputLogits: predictionLogits, h: nextH, c: nextC } = model.forwardStep(inputTensor, h, c);
      h = nextH; // Обновляем состояния для следующего шага
      c = nextC;

      // Применяем Softmax к логитам, чтобы получить распределение вероятностей
      const predictionProbs = softmax(predictionLogits);

      // Выбираем следующее слово на основе наибольшей вероятности (жадный подход)
      currentWord = getWordFromPrediction(predictionProbs, indexToWord);
      generatedSequence.push(currentWord);

      // Добавим проверку на <unk> токен для более чистого вывода
      if (currentWord === '<unk>' && i < numWords - 1) {
          generatedSequence.push('(продолжение неизвестно)');
          break; // Прекращаем генерацию, если модель "застряла" на неизвестном токене
      }
    }
    setOutput(`Сгенерированный текст: ${generatedSequence.join(' ')}`);
    setStatus('Генерация текста завершена.');
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '800px', margin: 'auto' }}>
      <Link href="/"><button style={{ marginBottom: '20px', padding: '10px', cursor: 'pointer' }}>← Назад в чат</button></Link>
      <h1>WordWise.js - Ваш универсальный ИИ-конструктор в браузере</h1>
      <p>Эта версия **WordWise.js** представляет собой продвинутый фреймворк для обучения нейронных сетей с акцентом на обработку естественного языка. Она включает:</p>
      <ul>
        <li>**Универсальный `Tensor`**: Базовая единица данных с поддержкой операций и автоматического дифференцирования.</li>
        <li>**Слой `Embedding`**: Для эффективного представления слов в виде векторов.</li>
        <li>**Ячейка `LSTM`**: Фундаментальный блок для обработки последовательностей и запоминания контекста.</li>
        <li>**Пакетная обработка (Batching)**: Для ускорения и стабилизации процесса обучения.</li>
        <li>**Объединенная функция потерь `Softmax + CrossEntropy`**: Для численно стабильного обучения классификации следующего слова.</li>
      </ul>
      <p>Несмотря на значительные улучшения, помните, что это учебный проект. Реальные фреймворки, такие как TensorFlow.js, используют высокооптимизированные WebGL/WebAssembly реализации для скорости и более сложные алгоритмы для автоматического дифференцирования.</p>

      <hr style={{ margin: '20px 0' }} />

      <h2>Набор данных для обучения</h2>
      <p>Модель будет обучаться предсказывать следующее слово на основе следующего простого текстового корпуса:</p>
      <pre style={{ backgroundColor: '#e8e8e8', padding: '10px', borderRadius: '5px', overflowX: 'auto' }}>
        "{textCorpus}"
      </pre>

      <hr style={{ margin: '20px 0' }} />

      <h2>Управление WordWise.js</h2>
      <div style={{ marginBottom: '15px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button
          onClick={() => initializeWordWise(textCorpus)}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          1. Инициализировать WordWise.js
        </button>
        <button
          onClick={trainWordWise}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#2196F3', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          2. Начать обучение
        </button>
      </div>
      <p style={{ fontWeight: 'bold' }}>Статус: {status}</p>

      {lossHistory.length > 0 && (
        <>
          <h3>История потерь (Средняя Cross-Entropy)</h3>
          <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #ccc', padding: '10px', borderRadius: '5px', backgroundColor: '#f9f9f9', marginTop: '10px' }}>
            {lossHistory.map((loss, index) => (
              <div key={index} style={{ fontSize: '0.9em', color: '#333' }}>
                Эпоха {index + 1}: {loss.toFixed(6)}
              </div>
            ))}
          </div>
          <p style={{ marginTop: '10px', fontWeight: 'bold' }}>Итоговая средняя потеря: {lossHistory[lossHistory.length - 1]?.toFixed(6)}</p>
        </>
      )}

      <hr style={{ margin: '20px 0' }} />

      <h2>Генерация текста</h2>
      <div style={{ marginBottom: '15px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button
          onClick={() => generateText("the", 10)}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#FF9800', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          Сгенерировать (начало: "the", 10 слов)
        </button>
        <button
          onClick={() => generateText("cat", 10)}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#FF9800', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          Сгенерировать (начало: "cat", 10 слов)
        </button>
        <button
          onClick={() => generateText("dog", 10)}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#FF9800', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          Сгенерировать (начало: "dog", 10 слов)
        </button>
      </div>
      <pre style={{ backgroundColor: '#e0e0e0', padding: '15px', borderRadius: '5px', overflowX: 'auto', whiteSpace: 'pre-wrap', minHeight: '80px', display: 'flex', alignItems: 'center' }}>
        {output || 'Нажмите кнопку "Сгенерировать", чтобы получить текст.'}
      </pre>
    </div>
  );
}
