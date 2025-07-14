import {genkit, Plugin} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';
import {firebasePlugin} from '@genkit-ai/firebase';

const plugins: Plugin<any>[] = [googleAI(), firebasePlugin()];

export const ai = genkit({
  plugins,
  flowStateStore: 'firebase',
  traceStore: 'firebase',
  enableTracingAndMetrics: true,
});
