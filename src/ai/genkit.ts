import {genkit, Plugin} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';
import {firebase} from '@genkit-ai/firebase';

const plugins: Plugin<any>[] = [googleAI(), firebase()];

export const ai = genkit({
  plugins,
  flowStateStore: 'firebase',
  traceStore: 'firebase',
  enableTracingAndMetrics: true,
});