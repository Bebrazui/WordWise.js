import {genkit} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';

// Note: The genkit CLI will search for a file named `genkit.conf.js`
// and load it automatically.

export const ai = genkit({
  plugins: [
    googleAI({
      apiVersion: ['v1', 'v1beta'],
    }),
  ],
  logSinks: [],
  enableTracing: true,
});
