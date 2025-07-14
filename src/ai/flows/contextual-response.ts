
'use server';
/**
 * @fileOverview A contextual response AI bot. This file now acts as a router.
 * It first determines the user's intent and then calls the appropriate flow.
 *
 * - contextualResponse - The main entry point function.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import {z} from 'zod';
import {ai} from '@/ai/genkit';
import {generateCode} from '@/ai/flows/code-generator';

import knowledgeBase from '@/data/knowledge-base.json';
import synonyms from '@/data/synonyms.json';
import wordConnections from '@/data/word-connections.json';
import learnedWords from '@/data/learned-words.json';
import morphology from '@/data/morphology.json';

// In a real-world scenario, this would be a proper database write.
// For this environment, we simulate it, but it won't persist across server restarts.
// NOTE: This is a simplified in-memory "database" for the sake of this example.
// In a real-world application, you would use a proper database like Firestore
// to persist learned data across server restarts and multiple instances.
const fs = {
  writeFile: (path: string, data: string, cb: (err?: any) => void) => {
    // This is a mock. In a real environment, you'd need Node's 'fs' module
    // and write permissions, which are not available in this context.
    console.log(`MOCK WRITE to ${path}: ${data.substring(0, 100)}...`);
    cb();
  },
};
import path from 'path';


// --- Type Definitions ---
type KnowledgeBase = {
  [intent: string]: {
    phrases: string[];
    responses: string[];
  };
};

type Synonyms = {
  [key: string]: string[];
};

type WordConnection = {
    value: string | string[];
    weight: number;
    type: string;
}

type WordConnections = {
  [word: string]: WordConnection[];
};

type LearnedWords = {
  [word:string]: string;
};

type Morphology = {
    prefixes: string[];
    suffixes: Record<string, string[]>;
    endings: string[];
};


// --- Load Knowledge Bases ---
const kb = knowledgeBase as KnowledgeBase;
const syn = synonyms as Synonyms;
const wc = wordConnections as WordConnections;
const lw = learnedWords as LearnedWords;
const morph = morphology as Morphology;


// --- State Management ---
const lastResponseMap = new Map<string, string>();
const sessionMemory = {
  lastUnknownWord: null as string | null,
};

// =================================================================
// === MAIN ROUTER AND ALGORITHM PIPELINE ==========================
// =================================================================


const ContextualResponseInputSchema = z.object({
  userInput: z
    .string()
    .describe('The user input to which the AI should respond.'),
  history: z
    .array(z.string())
    .optional()
    .describe('The last 3 messages in the conversation.'),
  experimental: z.boolean().optional().describe('Flag to enable experimental response generation.')
});
export type ContextualResponseInput = z.infer<
  typeof ContextualResponseInputSchema
>;

const ContextualResponseOutputSchema = z.object({
  aiResponse: z.string().describe('The AI response based on the user input.'),
});
export type ContextualResponseOutput = z.infer<
  typeof ContextualResponseOutputSchema
>;

// --- New Intent Router ---

const IntentSchema = z.object({
    intent: z.enum(['CONVERSATION', 'CODE_GENERATION'])
        .describe('The intent of the user. CONVERSATION for general chat, CODE_GENERATION for requests to write code.'),
});

const intentRouterPrompt = ai.definePrompt({
    name: 'intentRouter',
    input: { schema: z.object({ userInput: z.string() }) },
    output: { schema: IntentSchema },
    prompt: `You are an intent classification system. Determine if the user's request is a general conversation or a request to generate code.

    - If the user is asking to "write code", "create a function", "generate a script", "how to write...", "напиши код", "создай компонент", or similar, classify the intent as CODE_GENERATION.
    - For all other inputs, like greetings, questions, or general chat, classify the intent as CONVERSATION.

    User request: "{{userInput}}"`,
});

const intentRouterFlow = ai.defineFlow(
    {
        name: 'intentRouterFlow',
        inputSchema: z.object({ userInput: z.string() }),
        outputSchema: IntentSchema,
    },
    async (input) => {
        const { output } = await intentRouterPrompt(input);
        return output!;
    }
);

// --- (NEW) Generative AI Flow ---

const GenerativeInputSchema = z.object({
    userInput: z.string(),
    history: z.array(z.string()).optional(),
    normalizedInput: z.string(),
    matchedIntent: z.string().optional(),
    connectionResponse: z.string().optional(),
    fallbackAnalysis: z.string().optional(),
});

const generativeResponsePrompt = ai.definePrompt({
    name: 'generativeResponsePrompt',
    input: { schema: GenerativeInputSchema },
    output: { schema: ContextualResponseOutputSchema },
    prompt: `You are a friendly and intelligent conversational AI assistant named WordWise. Your goal is to provide helpful, natural-sounding responses in Russian.

    Current User Input: "{{userInput}}"
    Conversation History (most recent first):
    {{#each history}}
    - {{this}}
    {{/each}}

    Here is some context my internal algorithms have prepared for you:
    - Normalized User Input: "{{normalizedInput}}"
    {{#if matchedIntent}}
    - My best guess for the user's intent is: "{{matchedIntent}}".
    {{/if}}
    {{#if connectionResponse}}
    - I found this interesting connection: "{{connectionResponse}}".
    {{/if}}
    {{#if fallbackAnalysis}}
    - I couldn't find a direct match. My analysis suggests: "{{fallbackAnalysis}}".
    {{/if}}

    Based on all of this information, please formulate a single, coherent, and helpful response to the user. Do not act as a router or mention the internal algorithms. Just respond naturally as WordWise.`,
});

const generativeResponseFlow = ai.defineFlow({
    name: 'generativeResponseFlow',
    inputSchema: GenerativeInputSchema,
    outputSchema: ContextualResponseOutputSchema
}, async (input) => {
    const { output } = await generativeResponsePrompt(input);
    return output!;
});


// --- Pipeline Stage 1: Input Cleaner & Lemmatizer ---
function lemmatize(word: string): string {
    word = word.toLowerCase();

    // Check if the word is already a known base form or synonym
    const knownWords = new Set(Object.keys(syn).concat(...Object.values(syn).flat(), Object.keys(kb), Object.keys(wc), Object.keys(lw)));
    if (knownWords.has(word)) {
      // Find the base word if the current word is a synonym
      for(const key in syn){
        if(syn[key].includes(word)) return key;
      }
      return word;
    }


    const adjEndings = ['ый', 'ий', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'его', 'ого', 'ему', 'ому', 'ую', 'юю', 'ой', 'ей'];
    for (const ending of adjEndings) {
        if (word.endsWith(ending)) {
            let base = word.slice(0, -ending.length);
             if (knownWords.has(base + 'ий')) return base + 'ий';
             if (knownWords.has(base + 'ый')) return base + 'ый';
             if (knownWords.has(base + 'о')) return base + 'о';
             if (knownWords.has(base + 'а')) return base + 'а';
            return base + 'ий'; // Default to a common form
        }
    }
    const nounEndings = ['а', 'у', 'е', 'ы', 'ом', 'ам', 'ах', 'ой', 'ей', 'ю', 'ями', 'ами', 'и'];
    for (const ending of nounEndings) {
        if (word.length > 3 && word.endsWith(ending)) {
            let base = word.slice(0, -ending.length);
            if (knownWords.has(base)) return base;
        }
    }
    const verbEndings = ['ешь', 'ет', 'ем', 'ете', 'ут', 'ют', 'ишь', 'ит', 'им', 'ите', 'ат', 'ят', 'л', 'ла', 'ло', 'ли', 'ть', 'ти', 'ся', 'сь'];
    for (const ending of verbEndings) {
        if(word.endsWith(ending)) {
            let base = word.slice(0, -ending.length);
            if(knownWords.has(base + 'ть')) return base + 'ть';
            if(knownWords.has(base + 'ти')) return base + 'ти';
        }
    }
    return word;
}


function cleanAndNormalizeInput(userInput: string): string {
  if (!userInput) return '';
  const cleaned = userInput.toLowerCase().trim().replace(/[.,!?]/g, '');
  const words = cleaned.split(/\s+/);
  const lemmatizedWords = words.map(word => lemmatize(word));
  return lemmatizedWords.join(' ');
}

function levenshteinDistance(a: string, b: string): number {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    const matrix = Array.from(Array(a.length + 1), () => Array(b.length + 1).fill(0));
    for (let i = 0; i <= a.length; i++) { matrix[i][0] = i; }
    for (let j = 0; j <= b.length; j++) { matrix[0][j] = j; }
    for (let i = 1; i <= a.length; i++) {
        for (let j = 1; j <= b.length; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            );
        }
    }
    return matrix[a.length][b.length];
}

// --- Pipeline Stage 2: Learning Algorithm ---
function checkForDefinition(userInput: string): string | null {
  if (!sessionMemory.lastUnknownWord) {
    return null;
  }
  const lowerInput = userInput.toLowerCase();
  const patterns = [
    new RegExp(`^${sessionMemory.lastUnknownWord}\\s*-\\s*(.+)`),
    new RegExp(`^${sessionMemory.lastUnknownWord}\\s*это\\s*(.+)`),
    new RegExp(`^${sessionMemory.lastUnknownWord}\\s*значит\\s*(.+)`),
  ];
  for (const pattern of patterns) {
    const match = lowerInput.match(pattern);
    if (match && match[1]) {
      return match[1].trim();
    }
  }
  return null;
}

async function saveLearnedWord(word: string, definition: string): Promise<void> {
  const filePath = path.join(process.cwd(), 'src', 'data', 'learned-words.json');
  const updatedLearnedWords = {...lw, [word]: definition};
  try {
    await new Promise<void>((resolve, reject) => {
      fs.writeFile(
        filePath,
        JSON.stringify(updatedLearnedWords, null, 2),
        err => {
          if (err) {
            console.error('Failed to write to learned-words.json', err);
            reject(err);
          } else {
            Object.assign(lw, updatedLearnedWords);
            resolve();
          }
        }
      );
    });
  } catch (error) {
    // Error is already logged
  }
}

// --- Pipeline Stage 3: Direct Response Handler ---
function handleDirectResponse(normalizedInput: string, history: string[]): string | null {
    if (history.length === 0) return null;
    const lastBotMessage = history[history.length - 1].toLowerCase();
    if (lastBotMessage.includes("я правильно тебя понял?")) {
        if (normalizedInput === "нет" || normalizedInput === "неправильно") {
            return "Понял, моя ошибка. Попробуй, пожалуйста, перефразировать свой вопрос, чтобы я лучше понял.";
        }
        if (normalizedInput === "да" || normalizedInput === "правильно" || normalizedInput === "ага") {
            return "Отлично! Рад, что мы на одной волне.";
        }
    }
    return null;
}

// --- Pipeline Stage 4: Contextual Analyzer ---
const questionAboutWellbeing = [
  'как дела', 'как ты', 'как поживаешь', 'как твое', 'как сам', 'как настроение', 'что нового', 'как самочувствие', 'как твои дела'
];
const positiveUserStates = [
  'хорошо', 'нормально', 'отлично', 'замечательно', 'прекрасно', 'порядок', 'ничего', 'пойдет', 'хороший', 'у меня все хорошо', 'все хорошо', 'все отлично', 'все прекрасно'
];
function handleWellbeingResponse(normalizedInput: string, history: string[]): string | null {
    if (history.length === 0) return null;
    const lastBotMessage = history[history.length - 1].toLowerCase();
    const wasAsked = questionAboutWellbeing.some(q => lastBotMessage.includes(lemmatize(q)));
    if (!wasAsked) return null;
    const isPositiveResponse = positiveUserStates.some(state => normalizedInput.includes(state));
    if (isPositiveResponse) {
        const suitableResponses = [
            'Рад это слышать!', 'Отлично! Чем теперь займемся?', 'Здорово! Если что-то понадобится, я здесь.', 'Это хорошо. Что дальше по плану?',
        ];
        let responses = suitableResponses;
        const lastResponse = lastResponseMap.get('wellbeing_reply');
        if (lastResponse && responses.length > 1) {
            responses = responses.filter(r => r !== lastResponse);
        }
        const response = responses[Math.floor(Math.random() * responses.length)];
        lastResponseMap.set('wellbeing_reply', response);
        return response;
    }
    return null;
}

// --- Pipeline Stage 5: Knowledge Base Matcher (The Strategist) ---
function findBestIntentMatch(normalizedInput: string): { intent: string; score: number } | null {
  let bestMatch: { intent: string; score: number } | null = null;
  for (const intent in kb) {
    if (intent === 'unknown_phrase' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;
    const intentData = kb[intent];
    for (const phrase of intentData.phrases) {
      const lowerPhrase = phrase.toLowerCase();
      if (normalizedInput.includes(lowerPhrase) && lowerPhrase.length > 0) {
        const score = lowerPhrase.length / normalizedInput.length;
        if (score > (bestMatch?.score ?? 0)) {
          bestMatch = { intent, score };
        }
      } 
      else {
        const distance = levenshteinDistance(normalizedInput, lowerPhrase);
        const threshold = Math.floor(lowerPhrase.length * 0.4);
        if (distance <= threshold && lowerPhrase.length > 3) {
          const score = (1 - distance / lowerPhrase.length) * 0.9;
          if (score > (bestMatch?.score ?? 0)) {
            bestMatch = { intent, score };
          }
        }
      }
    }
  }
  return bestMatch;
}


// --- Pipeline Stage 6: Creative Connection Algorithm (The Quantum Generator) ---
function generateConnectionResponse(normalizedInput: string): string | null {
    const words = [...new Set(normalizedInput.split(/\s+/).filter(w => w.length > 2))];
    if (words.length === 0) return null;
    const foundFacts: { fact: string; weight: number }[] = [];
    const allWordsAndSynonyms = new Set<string>();
    words.forEach(word => {
        allWordsAndSynonyms.add(word);
        const baseWord = syn[word] ? word : Object.keys(syn).find(key => syn[key].includes(word));
        if (baseWord && syn[baseWord]) {
            syn[baseWord].forEach(s => allWordsAndSynonyms.add(s));
        }
    });
    for (const word of allWordsAndSynonyms) {
        if (wc[word]) {
            const connections = [...wc[word]].sort((a, b) => b.weight - a.weight);
            let fact = `${word.charAt(0).toUpperCase() + word.slice(1)}`;
            let hasFact = false;
            let totalWeight = 0;
            for (const conn of connections.slice(0, 2)) {
                const connValue = Array.isArray(conn.value) ? conn.value.join(', ') : conn.value;
                let newPart = '';
                switch(conn.type) {
                    case 'is_a': newPart = ` - это ${connValue}`; break;
                    case 'can_do': newPart = `${hasFact ? ', который' : ''} может ${connValue}`; break;
                    case 'related_to': newPart = `. Связано с: ${connValue}`; break;
                    case 'property_of': newPart = `. Является свойством: ${connValue}`; break;
                    case 'antonym': newPart = `, в противоположность - ${connValue}`; break;
                }
                if (newPart) {
                    fact += newPart;
                    hasFact = true;
                    totalWeight += conn.weight;
                }
            }
            if (hasFact) {
                fact += '.';
                foundFacts.push({ fact, weight: totalWeight });
            }
        }
    }
    if (foundFacts.length === 0) return null;
    foundFacts.sort((a, b) => b.weight - a.weight);
    if (foundFacts.length === 1) return foundFacts[0].fact;
    if (foundFacts.length > 1) {
        const fact1 = foundFacts[0].fact.replace(/\.$/, '');
        const fact2 = foundFacts[1].fact.toLowerCase();
        return `${fact1}, что, в свою очередь, связано с понятием "${fact2.split(' ')[0]}". Это интересная взаимосвязь.`;
    }
    return null;
}

// --- Pipeline Stage 7: Observer & Smart Fallback ---
function getFallbackResponse(normalizedInput: string): string {
    const wordsInInput = normalizedInput.split(/\s+/).filter(w => w);
    let unknownWord = wordsInInput.find(w => 
        w.length > 2 && 
        !syn[w] && 
        !Object.values(syn).flat().includes(w) && // Also check if it's a known synonym value
        !Object.keys(kb).some(k => kb[k].phrases.includes(w)) && 
        !lw[w as keyof typeof lw]
    );
    if (!unknownWord && wordsInInput.length > 0) {
        unknownWord = wordsInInput
            .filter(w => w.length > 2)
            .sort((a, b) => b.length - a.length)[0];
    } 
    if (!unknownWord) {
        return kb.unknown_phrase.responses[Math.floor(Math.random() * kb.unknown_phrase.responses.length)];
    }
    const thoughtfulResponses = [
        `Я размышляю над словом "${unknownWord}"... и что оно может означать в этом контексте. Можешь объяснить? Например: ${unknownWord} - это...`,
        'Это сложный вопрос. Мои алгоритмы ищут наиболее подходящий ответ, но пока безуспешно. Попробуешь перефразировать?',
        `Хм, слово "${unknownWord}" поставило меня в тупик. Не мог бы ты его пояснить?`,
    ];
    let responses = thoughtfulResponses;
    const lastResponse = lastResponseMap.get('fallback');
    if (lastResponse && responses.length > 1) {
        responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set('fallback', response);
    const mentionedWord = response.match(/"(.*?)"/);
    if (mentionedWord && mentionedWord[1]) {
        sessionMemory.lastUnknownWord = mentionedWord[1];
    } else {
        sessionMemory.lastUnknownWord = null;
    }
    return response;
}

function synonymize(sentence: string): string {
  const words = sentence.split(/(\s+|,|\.|\?|!)/);
  const newWords = words.map(word => {
    const lowerWord = word.toLowerCase();
    const synonymList = syn[lowerWord];
    if (synonymList && Math.random() < 0.75) { 
      const randomSynonym = synonymList[Math.floor(Math.random() * synonymList.length)];
      if (word.length > 0 && word[0] === word[0].toUpperCase()) {
        return randomSynonym.charAt(0).toUpperCase() + randomSynonym.slice(1);
      }
      return randomSynonym;
    }
    return word;
  });
  return newWords.join('');
}


async function generateConversationalResponse(
  userInput: string,
  history: string[] = [],
  experimental = false
): Promise<string> {
  const normalizedInput = cleanAndNormalizeInput(userInput);
  if (!normalizedInput) {
      return "Пожалуйста, скажи что-нибудь.";
  }

  // --- New Generative AI Path (if experimental mode is on) ---
  if (experimental) {
      const bestMatch = findBestIntentMatch(normalizedInput);
      const connectionResponse = generateConnectionResponse(normalizedInput);
      const fallbackResponse = getFallbackResponse(normalizedInput);
      
      const generativeInput: z.infer<typeof GenerativeInputSchema> = {
          userInput,
          history,
          normalizedInput,
          matchedIntent: bestMatch?.intent,
          connectionResponse: connectionResponse ?? undefined,
          fallbackAnalysis: !bestMatch && !connectionResponse ? fallbackResponse : undefined,
      };

      const result = await generativeResponseFlow(generativeInput);
      return result.aiResponse;
  }

  // --- Original Logic Path ---
  const definition = checkForDefinition(userInput);
  if (definition && sessionMemory.lastUnknownWord) {
    const learnedWord = sessionMemory.lastUnknownWord;
    await saveLearnedWord(learnedWord, definition);
    sessionMemory.lastUnknownWord = null;
    return `Понял! Теперь я знаю, что ${learnedWord} - это ${definition}. Спасибо!`;
  }
  if (sessionMemory.lastUnknownWord && !userInput.toLowerCase().startsWith(sessionMemory.lastUnknownWord)) {
      sessionMemory.lastUnknownWord = null;
  }
  const directResponse = handleDirectResponse(normalizedInput, history);
  if (directResponse) {
      return directResponse;
  }
  const wellbeingResponse = handleWellbeingResponse(normalizedInput, history);
  if (wellbeingResponse) {
      return synonymize(wellbeingResponse);
  }
  
  const bestMatch = findBestIntentMatch(normalizedInput);
  if (bestMatch && bestMatch.score > 0.6) {
    let responses = kb[bestMatch.intent].responses;
    const lastResponse = lastResponseMap.get(bestMatch.intent);
    if (lastResponse && responses.length > 1) {
      responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set(bestMatch.intent, response);
    if (bestMatch.score < 0.95 && bestMatch.score > 0.6) {
      return synonymize(response) + ' Я правильно тебя понял?';
    }
    return synonymize(response);
  }
  const connectionResponse = generateConnectionResponse(normalizedInput);
  if (connectionResponse) {
    return synonymize(connectionResponse);
  }
  const learnedMeaning = lw[normalizedInput as keyof typeof lw] || lw[lemmatize(normalizedInput) as keyof typeof lw];
  if (learnedMeaning) {
    const lemmatizedWord = lemmatize(normalizedInput);
    return `Я помню, ты говорил, что ${lemmatizedWord} - это ${learnedMeaning}. Хочешь поговорить об этом?`;
  }
  const fallbackResponse = getFallbackResponse(normalizedInput);
  return synonymize(fallbackResponse);
}


// --- Main Entry Point ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  const { intent } = await intentRouterFlow({ userInput: input.userInput });

  let aiResponse: string;

  if (intent === 'CODE_GENERATION') {
    const codeResult = await generateCode({ prompt: input.userInput });
    aiResponse = codeResult.code;
  } else {
    // Standard conversational flow
    const history = input.history || [];
    aiResponse = await generateConversationalResponse(input.userInput, history, input.experimental);
  }

  return {aiResponse};
}
