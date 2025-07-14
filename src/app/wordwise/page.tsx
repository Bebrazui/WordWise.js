
"use client";

import React, { useState, useRef, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Info, Upload, Download, Settings, TestTube2, CheckCircle, ImagePlus, FileText } from 'lucide-react';

import { serializeModel, deserializeModel, WordWiseModel, ImageWiseModel, BaseModel, VocabData } from '../../lib/model';
import { getWordFromPrediction } from '../../utils/tokenizer';
import { imageToTensor } from '../../utils/image-processor';
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

type TrainingImageItem = {
    file: File;
    previewUrl: string;
    label: string;
};

type TrainingDataType = {
  type: 'text';
  corpus: string;
} | {
  type: 'image';
  items: TrainingImageItem[];
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

  // Image-specific params
  const [imageSize, setImageSize] = useState(32);


  const { setTrainedModel, setVocabData, temperature, setTemperature } = useTrainedModel();
  const { toast } = useToast();
  
  const workerRef = useRef<Worker | null>(null);
  const modelRef = useRef<BaseModel | null>(null);
  const vocabDataRef = useRef<VocabData | null>(null);

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
          if (payload.sampleWords) {
            setSampleWords(payload.sampleWords);
          }
          let initMsg = `Модель (${payload.type}) инициализирована.`;
          if (payload.vocabSize) initMsg += ` Размер словаря: ${payload.vocabSize}.`;
          if (payload.numClasses) initMsg += ` Количество классов: ${payload.numClasses}.`;
          setOutput(initMsg);
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
          console.error("Worker Error:", payload.message);
          setIsTraining(false);
          break;
      }
    };

    return () => {
      worker.terminate();
    };
  }, [setTrainedModel, toast]);


  const initializeWordWise = useCallback(async () => {
    setStatus('Инициализация...');
    if (trainingData.type === 'text') {
        workerRef.current?.postMessage({
            type: 'initialize',
            payload: {
                type: 'text',
                textCorpus: trainingData.corpus,
                embeddingDim,
                hiddenSize,
                learningRate
            }
        });
    } else if (trainingData.type === 'image') {
        if(trainingData.items.length < 2) {
            toast({ title: "Ошибка", description: "Для обучения нужно как минимум 2 изображения.", variant: "destructive" });
            return;
        }
        setStatus('Обработка изображений...');
        
        try {
            const processedItems = await Promise.all(
                trainingData.items.map(async (item) => {
                    // This function now returns the raw pixel data and shape
                    const { pixelData, shape } = await imageToTensor(URL.createObjectURL(item.file), imageSize, imageSize);
                    return {
                        pixelData, // Float32Array
                        shape,     // [channels, height, width]
                        label: item.label
                    };
                })
            );

            // Transferable objects for efficiency
            const transferables = processedItems.map(item => item.pixelData.buffer);

            setStatus('Инициализация модели в Worker...');
            workerRef.current?.postMessage({
                type: 'initialize',
                payload: {
                    type: 'image',
                    items: processedItems, // Send processed data
                    imageSize,
                    learningRate
                }
            }, transferables);

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : "Неизвестная ошибка обработки изображений.";
            setStatus(`Ошибка: ${errorMessage}`);
            toast({ title: "Ошибка", description: errorMessage, variant: "destructive" });
        }
    }
  }, [trainingData, embeddingDim, hiddenSize, learningRate, imageSize, toast]);
  
  const stopTraining = () => {
    workerRef.current?.postMessage({ type: 'stop' });
  };

  const trainWordWise = async () => {
    if (!isInitialized || isTraining) return;
    setIsTraining(true);
    setIsTrained(false);
    setStatus('Начинается обучение...');
    setTrainingProgress(0);

    let payload: any;
     if (trainingData.type === 'text') {
        payload = {
            type: 'text',
            numEpochs,
            learningRate,
            batchSize
        };
    } else if (trainingData.type === 'image') {
        payload = {
            type: 'image',
            numEpochs,
            learningRate,
            batchSize
        };
    } else {
        toast({ title: "Ошибка", description: "Неизвестный тип данных для обучения.", variant: "destructive" });
        setIsTraining(false);
        return;
    }

    workerRef.current?.postMessage({
      type: 'train',
      payload: { ...payload, lossHistory }
    });
  };

  const applyToChat = () => {
     if (!modelRef.current || !vocabDataRef.current) {
        toast({ title: "Ошибка", description: "Нет модели для применения. Обучите или загрузите модель.", variant: "destructive" });
        return;
    }
    if (modelRef.current.type !== 'text') {
        toast({ title: "Ошибка", description: "К чату можно применить только текстовую модель.", variant: "destructive" });
        return;
    }
    setTrainedModel(modelRef.current as WordWiseModel);
    setVocabData(vocabDataRef.current);
    toast({ title: "Успех!", description: "Модель успешно применена к чату. Можете вернуться и пообщаться." });
  };

  const generateText = (startWord: string) => {
    if (!modelRef.current || !vocabDataRef.current || modelRef.current.type !== 'text' || !('wordToIndex' in vocabDataRef.current)) {
      toast({ title: "Ошибка", description: "Сначала обучите или загрузите текстовую модель.", variant: "destructive" });
      return;
    }
     const { wordToIndex, indexToWord } = vocabDataRef.current;
     const textModel = modelRef.current as WordWiseModel;

     let currentWord = startWord.toLowerCase();
     
     if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>';
        setOutput(`Начальное слово "${startWord}" не найдено. Используем "<unk>".\n`);
     } else {
        setOutput('');
     }

     let generatedSequence = [currentWord];
     let {h0: h, c0: c} = textModel.initializeStates(1);

     setStatus(`Генерация текста, начало: "${currentWord}"...`);
     setLatestPredictions([]);

     const numWords = 10;
     for (let i = 0; i < numWords; i++) {
        const inputTensor = new Tensor([wordToIndex.get(currentWord) || 0], [1]);
        const { outputLogits: predictionLogits, h: nextH, c: nextC } = textModel.forward(inputTensor, h, c);
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

    const newImageItems: TrainingImageItem[] = Array.from(files).map(file => ({
        file,
        previewUrl: URL.createObjectURL(file),
        label: file.name.split('.').slice(0, -1).join('.') || 'untitled'
    }));

    setTrainingData(prev => {
        if (prev.type === 'image') {
            return { type: 'image', items: [...prev.items, ...newImageItems] };
        }
        return { type: 'image', items: newImageItems };
    });
     
    event.target.value = '';
  }

  const handleLabelChange = (index: number, newLabel: string) => {
    if (trainingData.type !== 'image') return;

    const updatedItems = [...trainingData.items];
    updatedItems[index].label = newLabel;
    setTrainingData({ type: 'image', items: updatedItems });
  };


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
      a.download = `wordwise-model-${modelRef.current.type}-${new Date().toISOString().slice(0,10)}.json`;
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
        
        if (loaded.model.type === 'text' && 'vocab' in loaded.vocabData) {
            setTrainingData({type: 'text', corpus: 'Загружена текстовая модель. Корпус не восстанавливается.'});
            const textModel = loaded.model as WordWiseModel;
            setEmbeddingDim(textModel.embeddingDim);
            setHiddenSize(textModel.hiddenSize);
            const wordsForSampling = loaded.vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
            const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
            setSampleWords(shuffled.slice(0, 4));
        } else if (loaded.model.type === 'image') {
            setTrainingData({type: 'image', items: []});
            toast({title: "Загружена модель изображений", description: "Данные для обучения не были восстановлены."});
        }
        
        setIsInitialized(true);
        setIsTrained(true);
        setStatus('Модель успешно загружена. Готова к дообучению или использованию.');
        setOutput('Модель загружена.');
        setLossHistory([]);

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
                <Tabs defaultValue="text" className="w-full" onValueChange={(value) => {
                    if (value === 'text') {
                        setTrainingData({ type: 'text', corpus: defaultCorpus });
                    } else {
                        setTrainingData({ type: 'image', items: [] });
                    }
                    setIsInitialized(false);
                    setIsTrained(false);
                    setStatus('Готов к инициализации.');
                }}>
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="text"><FileText className="w-4 h-4 mr-2"/>Текст</TabsTrigger>
                        <TabsTrigger value="image"><ImagePlus className="w-4 h-4 mr-2"/>Изображения</TabsTrigger>
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
                             <Button asChild variant="outline" size="sm" disabled={isInitialized || isTraining}>
                                <Label htmlFor="image-upload" className="cursor-pointer">
                                    Загрузить изображения
                                </Label>
                             </Button>
                            <Input id="image-upload" type="file" multiple accept="image/*" className="sr-only" onChange={handleImageUpload} disabled={isInitialized || isTraining} />
                            <p className="text-xs text-gray-500 mt-2">Загрузите изображения для обучения</p>
                        </div>
                         {trainingData.type === 'image' && trainingData.items.length > 0 && (
                            <div className="mt-4 space-y-3 max-h-64 overflow-y-auto">
                                <p className="text-sm font-medium">Загружено {trainingData.items.length} изображений:</p>
                                {trainingData.items.map((item, index) => (
                                    <div key={index} className="flex items-center gap-3 p-2 border rounded-md">
                                        <img src={item.previewUrl} alt={item.label} className="w-12 h-12 rounded-md object-cover" />
                                        <Input
                                            type="text"
                                            value={item.label}
                                            onChange={(e) => handleLabelChange(index, e.target.value)}
                                            className="flex-grow"
                                            disabled={isTraining || isInitialized}
                                        />
                                    </div>
                                ))}
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
                        {trainingData.type === 'text' ? (
                           <>
                               <div className="grid grid-cols-2 gap-4">
                                 <div>
                                    <Label htmlFor="embeddingDim">Размер эмбеддинга</Label>
                                    <Input id="embeddingDim" type="number" value={embeddingDim} onChange={e => setEmbeddingDim(parseInt(e.target.value, 10))} min="8" step="8" disabled={isTraining || isInitialized}/>
                                 </div>
                                 <div>
                                    <Label htmlFor="hiddenSize">Размер скрытого слоя</Label>
                                    <Input id="hiddenSize" type="number" value={hiddenSize} onChange={e => setHiddenSize(parseInt(e.target.value, 10))} min="16" step="16" disabled={isTraining || isInitialized}/>
                                 </div>
                               </div>
                           </>
                        ) : (
                           <>
                               <div>
                                  <Label htmlFor="imageSize">Размер изображения (px)</Label>
                                  <Input id="imageSize" type="number" value={imageSize} onChange={e => setImageSize(parseInt(e.target.value, 10))} min="16" step="8" disabled={isTraining || isInitialized}/>
                               </div>
                           </>
                        )}
                       <div className="grid grid-cols-2 gap-4">
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
                          disabled={isTraining || !isTrained || trainingData.type !== 'text'}
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
                        Чем ниже значение, тем точнее модель предсказывает следующее слово или классифицирует изображение.
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
