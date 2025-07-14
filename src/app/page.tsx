
import { Chat } from "@/components/chat";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Code } from "lucide-react";

export default function Home() {
  return (
    <>
      <div className="absolute top-4 right-24 md:right-32 z-10">
        <Button asChild variant="outline">
          <Link href="/code-assistant">
            <Code className="mr-2 h-4 w-4" />
            Code Assistant
          </Link>
        </Button>
      </div>
      <Chat />
    </>
  );
}
