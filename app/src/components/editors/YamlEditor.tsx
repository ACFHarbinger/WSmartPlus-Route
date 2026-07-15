import Editor from "@monaco-editor/react";
import { useAppStore } from "../../store/app";

interface Props {
  value: string;
  onChange: (value: string) => void;
}

/** Monaco YAML editor for Config Editor raw mode (§G.13). */
export default function YamlEditor({ value, onChange }: Props) {
  const theme = useAppStore((s) => s.effectiveTheme);

  return (
    <div className="rounded-lg border border-canvas-border overflow-hidden">
      <Editor
        height="60vh"
        language="yaml"
        theme={theme === "dark" ? "vs-dark" : "light"}
        value={value}
        onChange={(v) => onChange(v ?? "")}
        options={{
          minimap: { enabled: false },
          fontSize: 12,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          lineNumbers: "on",
          wordWrap: "on",
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 2,
          renderWhitespace: "selection",
        }}
      />
    </div>
  );
}
