
"use client";
import { useState, useRef, useEffect, FormEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Settings, BrainCircuit, Bot } from 'lucide-react';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import Link from 'next/link';

// Определяем тип сообщения прямо здесь
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExperimentalMode, setIsExperimentalMode] = useState(false);

  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const modelName = "WordWise.js Engine";

  useEffect(() => {
    if (scrollAreaRef.current) {
        const viewport = scrollAreaRef.current.querySelector('div');
        if (viewport) {
            viewport.scrollTop = viewport.scrollHeight;
        }
    }
  }, [messages]);


  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Имитация ответа от локальной модели
    setTimeout(() => {
        const botResponse: ChatMessage = {
            role: 'assistant',
            content: `Я локальная модель и получил ваше сообщение: "${userMessage.content}". Моя логика ответа пока не подключена к этому интерфейсу.`
        };
        setMessages(prev => [...prev, botResponse]);
        setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100 dark:bg-gray-900">
      <Card className="w-full max-w-2xl h-[90vh] flex flex-col shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between border-b p-4">
          <div className="flex items-center gap-3">
            <Bot className="w-8 h-8 text-primary" />
            <div>
              <CardTitle className="text-lg">{modelName}</CardTitle>
              <p className="text-xs text-muted-foreground">Прототип ИИ-ассистента</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/wordwise" passHref>
              <Button variant="ghost" size="icon" aria-label="WordWise.js Training">
                <BrainCircuit className="w-5 h-5" />
              </Button>
            </Link>
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" aria-label="Настройки">
                  <Settings className="w-5 h-5" />
                </Button>
              </SheetTrigger>
              <SheetContent>
                <SheetHeader>
                  <SheetTitle>Настройки</SheetTitle>
                </SheetHeader>
                <div className="py-4">
                  <div className="flex items-center justify-between rounded-lg border p-3 shadow-sm">
                    <div className="space-y-0.5">
                      <Label>Генеративный режим</Label>
                      <p className="text-[0.8rem] text-muted-foreground">
                        Включает ИИ-генерацию ответов вместо шаблонных. (Пока не активно)
                      </p>
                    </div>
                    <Switch
                      checked={isExperimentalMode}
                      onCheckedChange={setIsExperimentalMode}
                      disabled={true}
                    />
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </CardHeader>
        <CardContent className="flex-grow p-0">
          <ScrollArea className="h-full" ref={scrollAreaRef}>
             <div className="p-4 space-y-4">
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
