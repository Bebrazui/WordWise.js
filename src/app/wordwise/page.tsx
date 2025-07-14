
// src/app/wordwise/page.tsx
"use client";

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Info } from 'lucide-react';

import { Tensor } from '../../lib/tensor';
import { crossEntropyLossWithSoftmaxGrad, softmax } from '../../lib/layers';
import { SGD } from '../../lib/optimizer';
import { WordWiseModel } from '../../lib/model';
import { buildVocabulary, wordsToInputTensors, wordsToTargetTensors, getWordFromPrediction, createBatches } from '../../utils/tokenizer';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';


import { useTrainedModel } from '@/hooks/use-trained-model';


export default function WordwisePage() {
  const [output, setOutput] = useState<string>('');
  const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number}[]>([]);
  const [status, setStatus] = useState<string>('Готов к инициализации.');
  const [isTraining, setIsTraining] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [textCorpus, setTextCorpus] = useState("вопрос: привет ответ: привет как дела\nвопрос: как дела ответ: все хорошо спасибо");
  const [sampleWords, setSampleWords] = useState<string[]>([]);
  
  // Параметры обучения
  const [learningRate, setLearningRate] = useState(0.01);
  const [numEpochs, setNumEpochs] = useState(500);

  const { setTrainedModel, setVocabData } = useTrainedModel();
  
  const modelRef = useRef<WordWiseModel | null>(null);
  const optimizerRef = useRef<SGD | null>(null);
  const vocabDataRef = useRef<{ vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } | null>(null);
  const trainingStopFlag = useRef(false);

  // Параметры модели
  const embeddingDim = 32;
  const hiddenSize = 64;
  const batchSize = 4;
  
  const initializeWordWise = () => {
    try {
      setStatus('Инициализация WordWise.js...');
      const { vocab, wordToIndex, indexToWord, vocabSize } = buildVocabulary(textCorpus);
      vocabDataRef.current = { vocab, wordToIndex, indexToWord, vocabSize };

      modelRef.current = new WordWiseModel(vocabSize, embeddingDim, hiddenSize);
      optimizerRef.current = new SGD(learningRate);
      
      setVocabData({ vocab, wordToIndex, indexToWord, vocabSize });
      
      const wordsForSampling = vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
      const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
      setSampleWords(shuffled.slice(0, 4));

      setLossHistory([]);
      setOutput('WordWise.js инициализирован. Словарь создан. Готов к обучению.');
      console.log('Словарь:', vocab);
      setStatus('Готов к обучению.');
      setIsInitialized(true);
    } catch (error) {
      console.error("Ошибка инициализации:", error);
      setStatus(`Ошибка инициализации: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  
  const stopTraining = () => {
    trainingStopFlag.current = true;
  };

  const trainWordWise = async () => {
    if (!isInitialized) {
        setStatus('Сначала инициализируйте модель.');
        return;
    }
    if (isTraining) return;

    setIsTraining(true);
    trainingStopFlag.current = false;
    const model = modelRef.current!;
    // Пересоздаем оптимизатор с актуальной скоростью обучения
    const optimizer = new SGD(learningRate);
    optimizerRef.current = optimizer;

    const { wordToIndex, vocabSize } = vocabDataRef.current!;

    const words = textCorpus.toLowerCase().match(/[a-zA-Zа-яА-ЯёЁ]+/g) || [];
    if (words.length < 2) {
        setStatus('Недостаточно слов для обучения в корпусе.');
        setIsTraining(false);
        return;
    }

    const inputTensors = wordsToInputTensors(words.slice(0, -1), wordToIndex);
    const targetTensors = wordsToTargetTensors(words.slice(1), wordToIndex, vocabSize);
    const batches = createBatches(inputTensors, targetTensors, batchSize);
    
    const newLossHistory: {epoch: number, loss: number}[] = [];
    setLossHistory(newLossHistory);
    setStatus('Начинается обучение...');
    setOutput('');
    setTrainingProgress(0);

    for (let epoch = 0; epoch < numEpochs; epoch++) {
      if (trainingStopFlag.current) {
        setStatus(`Обучение остановлено на эпохе ${epoch + 1}.`);
        break;
      }
      let epochLoss = 0;
      let h = model.initializeStates(batchSize).h0;
      let c = model.initializeStates(batchSize).c0;

      for (const batch of batches) {
        const { outputLogits: predictionLogits, h: nextH, c: nextC } = model.forwardStep(batch.inputs, h, c);
        h = nextH.detach();
        c = nextC.detach();

        const lossTensor = crossEntropyLossWithSoftmaxGrad(predictionLogits, batch.targets);
        epochLoss += lossTensor.data[0];
        lossTensor.backward();
        optimizer.step(model.getParameters());
      }

      const avgEpochLoss = epochLoss / batches.length;
      newLossHistory.push({epoch: epoch + 1, loss: avgEpochLoss});
      
      if (epoch % 10 === 0 || epoch === numEpochs - 1) {
        setLossHistory([...newLossHistory]);
        setStatus(`Обучение: Эпоха ${epoch + 1}/${numEpochs}, Потеря: ${avgEpochLoss.toFixed(6)}`);
        setTrainingProgress(((epoch + 1) / numEpochs) * 100);
        await new Promise(resolve => setTimeout(resolve, 0)); 
      }
    }
    
    setTrainedModel(model);
    if (!trainingStopFlag.current) {
        setStatus('Обучение завершено. Модель готова к использованию в чате!');
        setOutput('Обучение WordWise.js завершено. Теперь можно вернуться в чат и пообщаться с обученной моделью!');
    }
    setIsTraining(false);
    trainingStopFlag.current = false;
  };

  const generateText = (startWord: string, numWords: number) => {
    if (!modelRef.current || !vocabDataRef.current) {
      setOutput('Модель не обучена. Сначала инициализируйте и обучите её.');
      setStatus('Генерация невозможна: модель не обучена.');
      return;
    }

    const model = modelRef.current;
    const { wordToIndex, indexToWord } = vocabDataRef.current;
    let currentWord = startWord.toLowerCase();
    
    if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>';
        setOutput(`Начальное слово "${startWord}" не найдено. Используем "<unk>".`);
    } else {
        setOutput('');
    }

    let generatedSequence = [currentWord];
    let h = model.initializeStates(1).h0;
    let c = model.initializeStates(1).c0;

    setStatus(`Генерация текста, начало: "${currentWord}"...`);

    for (let i = 0; i < numWords; i++) {
      const inputTensor = new Tensor([wordToIndex.get(currentWord) || 0], [1]);
      const { outputLogits: predictionLogits, h: nextH, c: nextC } = model.forwardStep(inputTensor, h, c);
      h = nextH;
      c = nextC;

      const predictionProbs = softmax(predictionLogits);
      currentWord = getWordFromPrediction(predictionProbs, indexToWord);
      
      if (currentWord === 'вопрос' || currentWord === 'ответ') continue;

      generatedSequence.push(currentWord);

      if (currentWord === '<unk>') {
          break;
      }
    }
    setOutput(prev => prev + `\nСгенерированный текст: ${generatedSequence.join(' ')}`);
    setStatus('Генерация текста завершена.');
  };

  return (
    <div className="container mx-auto p-4 md:p-8 bg-slate-50 min-h-screen">
       <div className="mb-8">
        <Button asChild variant="ghost">
          <Link href="/">← Назад в чат</Link>
        </Button>
      </div>
      
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">Тренажерный Зал WordWise.js</h1>
        <p className="text-lg text-muted-foreground mt-2">Здесь вы можете обучить свою собственную языковую модель</p>
      </header>
      
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Шаг 1: Подготовьте данные</CardTitle>
              <CardDescription>Введите текст для обучения. Чтобы научить модель диалогу, используйте формат "вопрос: ... ответ: ...".</CardDescription>
            </CardHeader>
            <CardContent>
              <Label htmlFor="corpus">Ваш обучающий корпус:</Label>
              <Textarea
                id="corpus"
                value={textCorpus}
                onChange={(e) => setTextCorpus(e.target.value)}
                placeholder="вопрос: привет ответ: привет как дела..."
                className="min-h-[150px] mt-2 font-mono"
                disabled={isTraining || isInitialized}
              />
               <Alert variant="default" className="mt-4">
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    Чем больше качественных примеров "вопрос-ответ", тем умнее будет модель.
                  </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Шаг 2: Настройте и обучите</CardTitle>
              <CardDescription>Задайте параметры обучения. Больше эпох — дольше, но качественнее обучение.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
               <div className="grid grid-cols-2 gap-4">
                 <div>
                    <Label htmlFor="epochs">Эпохи обучения</Label>
                    <Input id="epochs" type="number" value={numEpochs} onChange={e => setNumEpochs(parseInt(e.target.value, 10))} min="1" max="100000" disabled={isTraining}/>
                 </div>
                 <div>
                    <Label htmlFor="lr">Скорость обучения</Label>
                    <Input id="lr" type="number" value={learningRate} onChange={e => setLearningRate(parseFloat(e.target.value))} step="0.001" min="0.0001" disabled={isTraining}/>
                 </div>
               </div>

              <Button onClick={initializeWordWise} disabled={isInitialized || isTraining} className="w-full">
                Инициализировать / Сбросить
              </Button>
              {!isTraining ? (
                 <Button onClick={trainWordWise} disabled={!isInitialized || isTraining} className="w-full">
                    Начать обучение
                 </Button>
              ) : (
                <Button onClick={stopTraining} variant="destructive" className="w-full">
                    Остановить обучение
                 </Button>
              )}
               {isTraining && <Progress value={trainingProgress} className="w-full" />}
              <p className="text-sm text-center text-muted-foreground pt-2">Статус: {status}</p>
            </CardContent>
          </Card>

           <Card>
            <CardHeader>
              <CardTitle>Шаг 3: Проверьте генерацию</CardTitle>
               <CardDescription>Здесь можно быстро проверить, как модель генерирует текст. Кнопки обновятся после инициализации.</CardDescription>
            </Header>
            <CardContent className="space-y-4">
                 <div className="grid grid-cols-2 gap-2">
                    {sampleWords.length > 0 ? (
                      sampleWords.map(word => (
                        <Button
                          key={word}
                          variant="outline"
                          onClick={() => generateText(word, 10)}
                          disabled={isTraining || !isInitialized}
                        >
                          {word}...
                        </Button>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground col-span-2 text-center">Инициализируйте модель, чтобы увидеть примеры.</p>
                    )}
                 </div>
                 <div className="mt-4 p-4 bg-slate-100 rounded-md min-h-[100px] text-gray-700 font-mono text-sm whitespace-pre-wrap">
                    {output || 'Результат генерации появится здесь...'}
                 </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-3">
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>График потерь (Loss)</CardTitle>
                    <CardDescription>
                        Этот график показывает, как "ошибка" модели уменьшается в процессе обучения.
                        Чем ниже значение, тем точнее модель предсказывает следующее слово.
                    </CardDescription>
                </CardHeader>
                <CardContent className="h-[400px] pr-8">
                     <ResponsiveContainer width="100%" height="100%">
                        {lossHistory.length > 0 ? (
                            <LineChart data={lossHistory} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" label={{ value: 'Эпоха', position: 'insideBottom', offset: -5 }}/>
                                <YAxis allowDecimals={false} domain={['dataMin', 'dataMax']} label={{ value: 'Потеря', angle: -90, position: 'insideLeft' }}/>
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '0.5rem' }}
                                    formatter={(value: number) => [value.toFixed(6), "Потеря"]}
                                />
                                <Legend verticalAlign="top" height={36} />
                                <Line type="monotone" dataKey="loss" name="Потеря при обучении" stroke="#8884d8" strokeWidth={2} dot={false} />
                            </LineChart>
                        ) : (
                            <div className="flex items-center justify-center h-full text-muted-foreground">
                                Начните обучение, чтобы увидеть график потерь.
                            </div>
                        )}
                    </ResponsiveContainer>
                </CardContent>
            </Card>
        </div>
      </div>
    </div>
  );
}
