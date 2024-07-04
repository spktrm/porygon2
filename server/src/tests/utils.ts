import { MessagePort } from "worker_threads";

export const port = {
    postMessage: () => {
        return;
    },
    onmessage: () => {},
    onmessageerror: () => {},
    close: () => {},
    start: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => {
        return true;
    },
    unref: () => {},
    ref: () => {},
    addListener: () => {},
    emit: () => {},
} as unknown as MessagePort;
