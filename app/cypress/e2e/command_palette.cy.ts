describe("Command palette & keyboard shortcuts (§D.7)", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.contains("Monitor").should("be.visible");
  });

  it("opens the command palette with Ctrl+K and navigates", () => {
    cy.get("body").type("{ctrl}k");
    cy.get("input[placeholder*='command' i], input[placeholder*='search' i]")
      .should("be.visible")
      .type("Settings");
    cy.contains("Commands", { matchCase: false }).should("be.visible");
    cy.focused().type("{enter}");
    cy.location("hash").should("contain", "m=settings");
  });

  it("closes the command palette with Escape", () => {
    cy.get("body").type("{ctrl}k");
    cy.get("input[placeholder*='command' i], input[placeholder*='search' i]").should("be.visible");
    cy.get("body").type("{esc}");
    cy.get("input[placeholder*='command' i], input[placeholder*='search' i]").should("not.exist");
  });
});
