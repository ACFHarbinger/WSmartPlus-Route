describe("Studio shell & navigation", () => {
  it("boots the app shell with the sidebar sections", () => {
    cy.visit("/");
    cy.contains("Monitor").should("be.visible");
    cy.contains("Analysis").should("be.visible");
    cy.contains("Launch").should("be.visible");
    cy.contains("Files").should("be.visible");
  });

  it("navigates between pages via the sidebar and syncs the URL hash", () => {
    cy.visit("/");
    cy.contains("Settings").click();
    cy.location("hash").should("contain", "m=settings");
    cy.contains("Report Studio").click();
    cy.location("hash").should("contain", "m=report_studio");
  });

  it("deep-links a page from the hash (useHashSync)", () => {
    cy.visit("/#m=process_monitor");
    cy.location("hash").should("contain", "m=process_monitor");
    cy.contains("Process Monitor").should("be.visible");
  });

  it("restores filter state encoded in the hash", () => {
    cy.visit("/#m=benchmark&l=1");
    cy.location("hash").should("contain", "m=benchmark");
    cy.location("hash").should("contain", "l=1");
  });
});
