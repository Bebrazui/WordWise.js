
"use client";
import { useState, useRef, useEffect, FormEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { BrainCircuit, Bot } from 'lucide-react';
import Link from 'next/link';

import { useTrainedModel } from '@/hooks/use-trained-model';
import { Tensor } from '@/lib/tensor';
import { softmax } from '@/lib/layers';
import { getWordFromPrediction } from '@/utils/tokenizer';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { trainedModel, vocabData } = useTrainedModel();

  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const modelName = trainedModel ? "WordWise.js (Обученная)" : "WordWise.js (Прототип)";

  useEffect(() => {
    if (scrollAreaRef.current) {
        const viewport = scrollAreaRef.current.querySelector('div');
        if (viewport) {
            viewport.scrollTop = viewport.scrollHeight;
        }
    }
  }, [messages]);

  const generateTextWithModel = (startWord: string, numWords: number): string => {
    if (!trainedModel || !vocabData) {
      return "Модель не обучена. Перейдите на страницу обучения.";
    }

    const { wordToIndex, indexToWord } = vocabData;
    let currentWord = startWord.toLowerCase().split(' ').pop() || '<unk>';

    if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>';
    }

    let generatedSequence = [startWord]; // Начинаем последовательность с полного ввода пользователя
    let h = trainedModel.initializeStates(1).h0;
    let c = trainedModel.initializeStates(1).c0;

    for (let i = 0; i < numWords; i++) {
      const inputTensor = new Tensor([wordToIndex.get(currentWord) || 0], [1]);
      const { outputLogits, h: nextH, c: nextC } = trainedModel.forwardStep(inputTensor, h, c);
      h = nextH;
      c = nextC;

      const predictionProbs = softmax(outputLogits);
      currentWord = getWordFromPrediction(predictionProbs, indexToWord);
      
      // Пропускаем служебные токены
      if (currentWord === 'вопрос' || currentWord === 'ответ') continue;

      generatedSequence.push(currentWord);

      if (currentWord === '<unk>') break;
    }
    // Убираем первое слово (которое было затравкой) и объединяем
    return generatedSequence.slice(1).join(' ');
  };


  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    // Имитация ответа от локальной модели
    setTimeout(() => {
        let botResponseContent: string;
        if (trainedModel && vocabData) {
            // Если модель обучена, генерируем ответ
            botResponseContent = generateTextWithModel(currentInput, 15);
        } else {
            // Если модель не обучена, даем шаблонный ответ
            botResponseContent = `Я прототип. Модель WordWise.js еще не обучена. Перейдите в раздел обучения, чтобы научить меня отвечать.`;
        }

        const botResponse: ChatMessage = {
            role: 'assistant',
            content: botResponseContent
        };
        setMessages(prev => [...prev, botResponse]);
        setIsLoading(false);
    }, 500);
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100 dark:bg-gray-900">
      <Card className="w-full max-w-2xl h-[90vh] flex flex-col shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between border-b p-4">
          <div className="flex items-center gap-3">
            <Bot className="w-8 h-8 text-primary" />
            <div>
              <CardTitle className="text-lg">{modelName}</CardTitle>
              <p className="text-xs text-muted-foreground">ИИ-ассистент на базе WordWise.js</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/wordwise" passHref>
              <Button variant="ghost" size="icon" aria-label="WordWise.js Training">
                <BrainCircuit className="w-5 h-5" />
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent className="flex-grow p-0">
          <ScrollArea className="h-full" ref={scrollAreaRef}>
             <div className="p-4 space-y-4">
               {messages.length === 0 && (
                 <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                    <p>Сообщений пока нет.</p>
                    <p className="text-sm">Чтобы начать, отправьте сообщение или обучите модель в разделе <BrainCircuit className="inline h-4 w-4" />.</p>
                 </div>
               )}
              {messages.map((message, index) => (
                <div key={index} className={`flex items-start gap-3 ${message.role === 'user' ? 'justify-end' : ''}`}>
                  {message.role === 'assistant' && (
                    <Avatar className="w-8 h-8">
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                  )}
                  <div className={`rounded-lg px-3 py-2 max-w-[80%] ${message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                    }`}>
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  </div>
                  {message.role === 'user' && (
                    <Avatar className="w-8 h-8">
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex items-start gap-3">
                  <Avatar className="w-8 h-8">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                  <div className="rounded-lg px-3 py-2 bg-muted">
                    <div className="flex items-center space-x-1">
                       <span className="h-2 w-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                       <span className="h-2 w-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                       <span className="h-2 w-2 bg-slate-400 rounded-full animate-bounce"></span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
        <div className="p-4 border-t bg-background">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Спросите что-нибудь..."
              autoComplete="off"
              disabled={isLoading}
              className="flex-grow"
            />
            <Button type="submit" disabled={isLoading}>
              Отправить
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
