import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KpiCard } from "../../src/components/ui/KpiCard";

describe("KpiCard", () => {
  it("renders label, formatted value and unit", () => {
    render(<KpiCard label="Total km" value={1234.567} unit="km" />);
    expect(screen.getByText("Total km")).toBeInTheDocument();
    expect(screen.getByText(/1,234\.57/)).toBeInTheDocument();
    expect(screen.getByText("km")).toBeInTheDocument();
  });

  it("shows an em dash for missing values", () => {
    render(<KpiCard label="Profit" value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("colours positive deltas green when higher is better", () => {
    const { container } = render(
      <KpiCard label="Profit" value={10} delta={2.5} colorize />
    );
    expect(container.querySelector(".kpi-delta-pos")).toHaveTextContent("+2.5");
    expect(container.querySelector(".kpi-delta-neg")).toBeNull();
  });

  it("colours positive deltas red when lower is better", () => {
    const { container } = render(
      <KpiCard label="Overflows" value={10} delta={2.5} colorize lowerIsBetter />
    );
    expect(container.querySelector(".kpi-delta-neg")).toHaveTextContent("+2.5");
  });

  it("hides the delta without colorize", () => {
    const { container } = render(<KpiCard label="Profit" value={10} delta={2.5} />);
    expect(container.querySelector(".kpi-delta-pos")).toBeNull();
  });
});
