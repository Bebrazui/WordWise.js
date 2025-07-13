import {genkit} from 'genkit';
import * as ai from '@genkit-ai/ai';
import {z} from 'zod';

export const aiInstance = genkit({
  plugins: [],
});

aiInstance.defineModel(
  {
    name: 'my-custom-model',
    configSchema: ai.GenerationCommonConfigSchema,
    info: {
      label: 'My Custom Model',
      supports: {
        media: false,
        output: ['text'],
        input: ['text'],
      },
    },
  },
  async (request, stream) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    const responseText = `This is a placeholder response for: ${request.candidates[0].message.content[0].text}`;
    if (stream) {
      stream.text(responseText);
    }
    return {
      candidates: [
        {
          index: 0,
          finishReason: 'stop',
          message: {
            role: 'model',
            content: [{text: responseText}],
          },
        },
      ],
    };
  }
);
