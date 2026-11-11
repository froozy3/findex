import { Container, getContainer } from "@cloudflare/containers";

export class Findex extends Container {
  defaultPort = 8000;
  sleepAfter = "15m";
}

interface Env {
  FINDEX: DurableObjectNamespace<Findex>;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return getContainer(env.FINDEX).fetch(request);
  },
};
