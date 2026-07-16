import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusPill } from "../../src/components/ui/StatusPill";

describe("StatusPill", () => {
  it.each([
    ["running", "Running"],
    ["completed", "Completed"],
    ["cancelled", "Cancelled"],
    ["failed", "Failed"],
  ] as const)("renders the %s status label", (status, label) => {
    render(<StatusPill status={status} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("shows the pulse dot only while running", () => {
    const { container: running } = render(<StatusPill status="running" />);
    expect(running.querySelector(".animate-pulse")).not.toBeNull();
    const { container: done } = render(<StatusPill status="completed" />);
    expect(done.querySelector(".animate-pulse")).toBeNull();
  });
});
