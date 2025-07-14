
"use client";

import React, { useState, useRef, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Info, Upload, Download, Settings, TestTube2, CheckCircle, ImagePlus, FileText } from 'lucide-react';

import { serializeModel, deserializeModel, WordWiseModel } from '../../lib/model';
import { getWordFromPrediction } from '../../utils/tokenizer';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Slider } from '@/components/ui/slider';
import { PredictionVisualizer, Prediction } from '@/components/ui/prediction-visualizer';
import { useToast } from '@/hooks/use-toast';
import { useTrainedModel } from '@/hooks/use-trained-model';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Tensor } from '@/lib/tensor';


const defaultCorpus = `вопрос: привет ответ: привет как твои дела
вопрос: как дела ответ: у меня все отлично спасибо а у тебя
вопрос: как твои дела ответ: неплохо спасибо что спросил
вопрос: чем занимаешься ответ: читаю интересную книгу о космосе
вопрос: какая сегодня погода ответ: сегодня солнечно и тепло прекрасный день
вопрос: что нового ответ: ничего особенного все по-старому
вопрос: у тебя есть хобби ответ: да я люблю программировать и создавать новое
вопрос: добрый день ответ: и вам добрый день`;

type TrainingDataType = {
  type: 'text';
  corpus: string;
} | {
  type: 'image';
  items: { dataUrl: string; label: string }[];
};


export default function WordwisePage() {
  const [output, setOutput] = useState<string>('');
  const [latestPredictions, setLatestPredictions] = useState<Prediction[]>([]);
  const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number}[]>([]);
  const [status, setStatus] = useState<string>('Готов к инициализации.');
  const [isTraining, setIsTraining] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isTrained, setIsTrained] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [sampleWords, setSampleWords] = useState<string[]>([]);

  // Training data state
  const [trainingData, setTrainingData] = useState<TrainingDataType>({ type: 'text', corpus: defaultCorpus });

  // Hyperparameters
  const [learningRate, setLearningRate] = useState(0.01);
  const [numEpochs, setNumEpochs] = useState(500);
  const [embeddingDim, setEmbeddingDim] = useState(64);
  const [hiddenSize, setHiddenSize] = useState(128);
  const [batchSize, setBatchSize] = useState(4);

  const { setTrainedModel, setVocabData, temperature, setTemperature } = useTrainedModel();
  const { toast } = useToast();
  
  const workerRef = useRef<Worker | null>(null);
  const modelRef = useRef<WordWiseModel | null>(null);
  const vocabDataRef = useRef<{ vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } | null>(null);

  // Setup Web Worker
  useEffect(() => {
    const worker = new Worker(new URL('./wordwise.worker.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const { type, payload } = event.data;
      switch (type) {
        case 'worker-ready':
          setStatus('Worker готов. Можно инициализировать модель.');
          break;
        case 'initialized':
          setStatus('Готов к обучению.');
          setIsInitialized(true);
          setIsTrained(false);
          setLossHistory([]);
          setLatestPredictions([]);
          setTrainedModel(null);
          setSampleWords(payload.sampleWords);
          setOutput(`WordWise.js инициализирован. Размер словаря: ${payload.vocabSize}.`);
          break;
        case 'progress':
          setLossHistory(prev => [...prev, { epoch: payload.epoch, loss: payload.loss }]);
          setStatus(`Обучение: Эпоха ${payload.epoch}, Потеря: ${payload.loss.toFixed(6)}`);
          setTrainingProgress(payload.progress);
          break;
        case 'training-complete':
          setStatus('Обучение завершено. Модель готова к проверке и применению.');
          setOutput('Обучение завершено. Проверьте генерацию и примените к чату.');
          setIsTraining(false);
          setIsTrained(true);
           try {
                const { model, vocabData } = deserializeModel(payload.modelJson);
                modelRef.current = model;
                vocabDataRef.current = vocabData;
                toast({ title: "Успех!", description: "Модель из воркера успешно загружена." });
            } catch (e) {
                toast({ title: "Ошибка", description: `Не удалось десериализовать модель из воркера: ${e}`, variant: "destructive" });
            }
          break;
        case 'training-stopped':
           setStatus(`Обучение остановлено на эпохе ${payload.epoch}.`);
           setIsTraining(false);
           break;
        case 'generation-result':
            setOutput(payload.text);
            setStatus('Генерация текста завершена.');
            break;
        case 'error':
          setStatus(`Ошибка в Worker: ${payload.message}`);
          setIsTraining(false);
          break;
      }
    };

    return () => {
      worker.terminate();
    };
  }, [setTrainedModel, toast]);


  const initializeWordWise = useCallback(() => {
    if (trainingData.type !== 'text') {
        toast({ title: "Ошибка", description: "Инициализация пока поддерживается только для текста.", variant: "destructive" });
        return;
    }
    setStatus('Инициализация WordWise.js...');
    workerRef.current?.postMessage({
        type: 'initialize',
        payload: {
            textCorpus: trainingData.corpus,
            embeddingDim,
            hiddenSize,
            learningRate
        }
    });
  }, [trainingData, embeddingDim, hiddenSize, learningRate, toast]);
  
  const stopTraining = () => {
    workerRef.current?.postMessage({ type: 'stop' });
  };

  const trainWordWise = async () => {
    if (trainingData.type !== 'text') {
        toast({ title: "Ошибка", description: "Обучение пока поддерживается только для текста.", variant: "destructive" });
        return;
    }
    if (!isInitialized || isTraining) return;

    setIsTraining(true);
    setIsTrained(false);
    setStatus('Начинается обучение...');
    setTrainingProgress(0);

    workerRef.current?.postMessage({
      type: 'train',
      payload: {
        textCorpus: trainingData.corpus,
        numEpochs,
        learningRate,
        batchSize,
        lossHistory
      }
    });
  };

  const applyToChat = () => {
     if (!modelRef.current || !vocabDataRef.current) {
        toast({ title: "Ошибка", description: "Нет модели для применения. Обучите или загрузите модель.", variant: "destructive" });
        return;
    }
    setTrainedModel(modelRef.current);
    setVocabData(vocabDataRef.current);
    toast({ title: "Успех!", description: "Модель успешно применена к чату. Можете вернуться и пообщаться." });
  };

  const generateText = (startWord: string) => {
    if (!modelRef.current || !vocabDataRef.current) {
      toast({ title: "Ошибка", description: "Сначала обучите или загрузите модель.", variant: "destructive" });
      return;
    }
     const { wordToIndex, indexToWord } = vocabDataRef.current!;
     let currentWord = startWord.toLowerCase();
     
     if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>';
        setOutput(`Начальное слово "${startWord}" не найдено. Используем "<unk>".\n`);
     } else {
        setOutput('');
     }

     let generatedSequence = [currentWord];
     let h = modelRef.current.initializeStates(1).h0;
     let c = modelRef.current.initializeStates(1).c0;

     setStatus(`Генерация текста, начало: "${currentWord}"...`);
     setLatestPredictions([]);

     const numWords = 10;
     for (let i = 0; i < numWords; i++) {
        const inputTensor = new Tensor([wordToIndex.get(currentWord) || 0], [1]);
        const { outputLogits: predictionLogits, h: nextH, c: nextC } = modelRef.current.forwardStep(inputTensor, h, c);
        h = nextH;
        c = nextC;

        const { chosenWord, topPredictions } = getWordFromPrediction(predictionLogits, indexToWord, temperature, generatedSequence);
        setLatestPredictions(topPredictions);
        
        if (chosenWord === 'вопрос' || chosenWord === 'ответ') continue;
        
        generatedSequence.push(chosenWord);
        currentWord = chosenWord;
        if (chosenWord === '<unk>') break;
     }
     setOutput(prev => prev + `Сгенерированный текст: ${generatedSequence.join(' ')}`);
     setStatus('Генерация текста завершена.');
  };
  
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const newImageItems: { dataUrl: string; label: string }[] = [];
    const filesArray = Array.from(files);

    filesArray.forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            newImageItems.push({
                dataUrl: e.target?.result as string,
                label: file.name.split('.')[0] || 'unknown'
            });
            if (newImageItems.length === filesArray.length) {
                setTrainingData({ type: 'image', items: newImageItems });
            }
        };
        reader.readAsDataURL(file);
    });
     event.target.value = '';
  }


  const handleSaveModel = () => {
    if (!modelRef.current || !vocabDataRef.current) {
      toast({ title: "Ошибка", description: "Нет модели для сохранения.", variant: "destructive" });
      return;
    }
    try {
      const modelJson = serializeModel(modelRef.current, vocabDataRef.current);
      const blob = new Blob([modelJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `wordwise-model-${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast({ title: "Успех", description: "Модель успешно сохранена." });
    } catch (error) {
      console.error("Ошибка сохранения модели:", error);
      toast({ title: "Ошибка сохранения", description: `${error instanceof Error ? error.message : 'Неизвестная ошибка'}`, variant: "destructive" });
    }
  };

  const handleLoadModel = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonContent = e.target?.result as string;
        const loaded = deserializeModel(jsonContent);

        modelRef.current = loaded.model;
        vocabDataRef.current = loaded.vocabData;
        
        setEmbeddingDim(loaded.model.embeddingDim);
        setHiddenSize(loaded.model.hiddenSize);

        setIsInitialized(true);
        setIsTrained(true);
        setStatus('Модель успешно загружена. Готова к дообучению или использованию.');
        setOutput('Модель загружена.');
        setLossHistory([]);

        const wordsForSampling = loaded.vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
        const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
        setSampleWords(shuffled.slice(0, 4));

        toast({ title: "Успех", description: "Модель успешно загружена." });
      } catch (error) {
        console.error("Ошибка загрузки модели:", error);
        toast({ title: "Ошибка загрузки", description: `${error instanceof Error ? error.message : 'Неверный формат файла'}`, variant: "destructive" });
        setStatus('Ошибка загрузки модели.');
      }
    };
    reader.readAsText(file);
    event.target.value = '';
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
        <p className="text-lg text-muted-foreground mt-2">Здесь вы можете создать, обучить и протестировать свою языковую или визуальную модель</p>
      </header>
      
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Шаг 1: Подготовьте данные</CardTitle>
              <CardDescription>Выберите тип данных для обучения: текст или изображения.</CardDescription>
            </CardHeader>
            <CardContent>
                <Tabs defaultValue="text" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="text" onClick={() => setTrainingData({ type: 'text', corpus: defaultCorpus })}><FileText className="w-4 h-4 mr-2"/>Текст</TabsTrigger>
                        <TabsTrigger value="image" onClick={() => setTrainingData({ type: 'image', items: [] })}><ImagePlus className="w-4 h-4 mr-2"/>Изображения</TabsTrigger>
                    </TabsList>
                    <TabsContent value="text" className="mt-4">
                        <Label htmlFor="corpus">Ваш обучающий корпус:</Label>
                        <Textarea
                            id="corpus"
                            value={trainingData.type === 'text' ? trainingData.corpus : ''}
                            onChange={(e) => setTrainingData({ type: 'text', corpus: e.target.value })}
                            placeholder="вопрос: привет ответ: привет как дела..."
                            className="min-h-[150px] mt-2 font-mono"
                            disabled={isTraining || isInitialized}
                        />
                        <Alert variant="default" className="mt-4">
                            <Info className="h-4 w-4" />
                            <AlertDescription>
                            Чем больше качественных примеров, тем умнее будет модель.
                            </AlertDescription>
                        </Alert>
                    </TabsContent>
                    <TabsContent value="image" className="mt-4">
                        <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6">
                            <ImagePlus className="w-12 h-12 text-gray-400 mb-2"/>
                            <Label htmlFor="image-upload" className="mb-2 text-center text-sm text-gray-600">Перетащите файлы сюда или нажмите для выбора</Label>
                             <Button asChild variant="outline" size="sm">
                                <Label htmlFor="image-upload" className="cursor-pointer">
                                    Загрузить изображения
                                </Label>
                             </Button>
                            <Input id="image-upload" type="file" multiple accept="image/*" className="sr-only" onChange={handleImageUpload}/>
                            <p className="text-xs text-gray-500 mt-2">Загрузите изображения для обучения</p>
                        </div>
                         {trainingData.type === 'image' && trainingData.items.length > 0 && (
                            <div className="mt-4">
                                <p className="text-sm font-medium">Загружено {trainingData.items.length} изображений:</p>
                                <div className="grid grid-cols-3 gap-2 mt-2 max-h-48 overflow-y-auto">
                                    {trainingData.items.map((item, index) => (
                                        <div key={index} className="relative group">
                                            <img src={item.dataUrl} alt={item.label} className="rounded-md object-cover aspect-square" />
                                            <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs text-center p-0.5 truncate">{item.label}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                         )}
                    </TabsContent>
                </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Шаг 2: Настройте и обучите</CardTitle>
              <CardDescription>Задайте параметры, инициализируйте модель и начните обучение.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="hyperparams">
                    <AccordionTrigger>
                        <div className="flex items-center gap-2">
                           <Settings className="h-5 w-5" />
                           <span>Параметры обучения и модели</span>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="space-y-4 pt-4">
                       <div className="grid grid-cols-2 gap-4">
                         <div>
                            <Label htmlFor="embeddingDim">Размер эмбеддинга</Label>
                            <Input id="embeddingDim" type="number" value={embeddingDim} onChange={e => setEmbeddingDim(parseInt(e.target.value, 10))} min="8" step="8" disabled={isTraining || isInitialized}/>
                         </div>
                         <div>
                            <Label htmlFor="hiddenSize">Размер скрытого слоя</Label>
                            <Input id="hiddenSize" type="number" value={hiddenSize} onChange={e => setHiddenSize(parseInt(e.target.value, 10))} min="16" step="16" disabled={isTraining || isInitialized}/>
                         </div>
                         <div>
                            <Label htmlFor="batchSize">Размер батча</Label>
                            <Input id="batchSize" type="number" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value, 10))} min="1" disabled={isTraining}/>
                         </div>
                         <div>
                            <Label htmlFor="lr">Скорость обучения</Label>
                            <Input id="lr" type="number" value={learningRate} onChange={e => setLearningRate(parseFloat(e.target.value))} step="0.001" min="0.0001" disabled={isTraining}/>
                         </div>
                       </div>
                       <div className="col-span-2">
                          <Label htmlFor="epochs">Эпохи обучения</Label>
                          <Input id="epochs" type="number" value={numEpochs} onChange={e => setNumEpochs(parseInt(e.target.value, 10))} min="1" max="100000" disabled={isTraining}/>
                       </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
                <Button onClick={initializeWordWise} disabled={isTraining} className="w-full">
                    Инициализировать / Сбросить модель
                </Button>
                <div className="flex gap-2">
                  {!isTraining ? (
                      <Button onClick={trainWordWise} disabled={!isInitialized || isTraining} className="w-full">
                          Начать/Продолжить обучение
                      </Button>
                  ) : (
                    <Button onClick={stopTraining} variant="destructive" className="w-full">
                        Остановить обучение
                    </Button>
                  )}
                  <Button onClick={applyToChat} disabled={!isTrained || isTraining} variant="secondary" className="w-full">
                    <CheckCircle className="mr-2 h-4 w-4" />
                    Применить к чату
                  </Button>
                </div>
                {isTraining && <Progress value={trainingProgress} className="w-full" />}
                <p className="text-sm text-center text-muted-foreground pt-2">Статус: {status}</p>
            </CardContent>
          </Card>
           <Card>
            <CardHeader>
              <CardTitle>Управление моделью</CardTitle>
              <CardDescription>Сохраните прогресс обучения или загрузите ранее сохраненную модель.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4">
               <Button onClick={handleSaveModel} disabled={isTraining || !isTrained} variant="outline">
                 <Download className="mr-2 h-4 w-4" /> Сохранить
               </Button>
               <Button asChild variant="outline" disabled={isTraining}>
                 <Label htmlFor="load-model-input" className="cursor-pointer flex justify-center items-center">
                    <Upload className="mr-2 h-4 w-4" /> Загрузить
                    <Input id="load-model-input" type="file" accept=".json" className="sr-only" onChange={handleLoadModel} />
                 </Label>
               </Button>
            </CardContent>
           </Card>

           <Card>
            <CardHeader>
              <CardTitle>Шаг 3: Проверьте генерацию</CardTitle>
               <CardDescription>Здесь можно быстро проверить, как модель генерирует текст и посмотреть "мысли" модели.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div>
                    <Label htmlFor="temperature">Температура генерации: {temperature.toFixed(2)}</Label>
                    <Slider
                        id="temperature"
                        min={0.1}
                        max={2.0}
                        step={0.05}
                        value={[temperature]}
                        onValueChange={(value) => setTemperature(value[0])}
                        disabled={isTraining}
                        className="mt-2"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Низкая: более предсказуемый текст. Высокая: более случайный.</p>
                </div>
                 <div className="grid grid-cols-2 gap-2">
                    {sampleWords.length > 0 ? (
                      sampleWords.map(word => (
                        <Button
                          key={word}
                          variant="outline"
                          onClick={() => generateText(word)}
                          disabled={isTraining || !isTrained}
                        >
                          <TestTube2 className="mr-2 h-4 w-4" />
                          {word}...
                        </Button>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground col-span-2 text-center">Инициализируйте или загрузите модель.</p>
                    )}
                 </div>
                 <div className="mt-4 p-4 bg-slate-100 rounded-md min-h-[100px] text-gray-700 font-mono text-sm whitespace-pre-wrap">
                    {output || 'Результат генерации появится здесь...'}
                 </div>
                 {latestPredictions.length > 0 && <PredictionVisualizer predictions={latestPredictions} />}
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
                                <XAxis dataKey="epoch" type="number" domain={['dataMin', 'dataMax']} label={{ value: 'Эпоха', position: 'insideBottom', offset: -5 }}/>
                                <YAxis allowDecimals={false} domain={['dataMin', 'auto']} label={{ value: 'Потеря', angle: -90, position: 'insideLeft' }}/>
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
