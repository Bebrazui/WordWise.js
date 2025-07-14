
'use server';
/**
 * @fileOverview A code generation AI flow.
 *
 * - generateCode - A function that handles the code generation process.
 * - GenerateCodeInput - The input type for the generateCode function.
 * - GenerateCodeOutput - The return type for the generateCode function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'zod';

const GenerateCodeInputSchema = z.object({
  prompt: z.string().describe('A detailed description of the code to be generated.'),
});
export type GenerateCodeInput = z.infer<typeof GenerateCodeInputSchema>;

const GenerateCodeOutputSchema = z.object({
  code: z.string().describe('The generated code snippet, formatted in a markdown block.'),
});
export type GenerateCodeOutput = z.infer<typeof GenerateCodeOutputSchema>;


const codeGenerationPrompt = ai.definePrompt({
    name: 'codeGenerationPrompt',
    input: { schema: GenerateCodeInputSchema },
    output: { schema: GenerateCodeOutputSchema },
    prompt: `You are an expert programmer specializing in TypeScript, React, and Next.js. Your task is to write a high-quality, production-ready code snippet based on the user's request.

The user wants to generate the following code:
"{{prompt}}"

Please generate the code. The code should be complete, correct, and enclosed in a single markdown code block (e.g., \`\`\`typescript ... \`\`\`). Do not add any explanatory text before or after the code block. Focus on creating clean, readable, and efficient code that follows best practices.
`,
});


const codeGeneratorFlow = ai.defineFlow(
  {
    name: 'codeGeneratorFlow',
    inputSchema: GenerateCodeInputSchema,
    outputSchema: GenerateCodeOutputSchema,
  },
  async (input) => {
    const { output } = await codeGenerationPrompt(input);
    return output!;
  }
);


export async function generateCode(
  input: GenerateCodeInput
): Promise<GenerateCodeOutput> {
    const response = await codeGeneratorFlow(input);
    return response;
}
