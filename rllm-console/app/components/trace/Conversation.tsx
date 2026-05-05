import { ArrowDown } from "lucide-react";

import type { ChatMessage } from "./MessageBubble";
import { MessageBubble } from "./MessageBubble";

interface Props {
  /** Prompt messages — the input to the LLM call. */
  messages: ChatMessage[] | null;
  /** Single assistant turn returned by the model. */
  response: ChatMessage | null;
}

/**
 * Renders one trace as a chat thread: the prompt messages first, then
 * a divider, then the assistant's response. When either side is
 * missing/malformed the caller falls back to the raw JSON view in
 * TraceCard, so this component just trusts what it gets.
 */
export function Conversation({ messages, response }: Props) {
  if (!messages?.length && !response) {
    return (
      <div className="text-xs italic text-faint">
        no messages on this trace
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {messages?.map((m, i) => (
        <MessageBubble key={`req-${i}`} message={m} index={i} />
      ))}
      {response && (
        <>
          <div className="flex items-center gap-2 px-1 py-0.5 text-[10px] uppercase tracking-wide text-faint">
            <ArrowDown className="h-3 w-3" />
            response
          </div>
          <MessageBubble message={response} />
        </>
      )}
    </div>
  );
}
