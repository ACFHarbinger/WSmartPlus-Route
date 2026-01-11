/*!
 * Circular sector utilities for geometric optimization.
 *
 * Used by the SWAP* operator to determine if two routes overlap spatially.
 */

/**
 * A circular sector defined by angular range [start, end].
 *
 * Angles are mapped to integers in [0, 65535] to avoid floating-point errors.
 *
 * # Example
 *
 * ```text
 * Depot at (0, 0)
 * Node 1 at (1, 0)  → angle = 0°     → 0
 * Node 2 at (0, 1)  → angle = 90°    → 16384
 * Node 3 at (-1, 0) → angle = 180°   → 32768
 * Node 4 at (0, -1) → angle = 270°   → 49152
 * ```
 */
#[derive(Clone, Copy, Default, Debug)]
pub struct CircleSector {
    pub start: i32,
    pub end: i32,
}

impl CircleSector {
    /** Creates a new sector containing a single point. */
    pub fn new(point: i32) -> Self {
        Self {
            start: point,
            end: point,
        }
    }

    /**
     * Computes positive modulo 65536.
     *
     * Ensures the result is always in [0, 65535].
     */
    fn positive_mod(i: i32) -> i32 {
        (i % 65536 + 65536) % 65536
    }

    /** Checks if a point is enclosed within this sector. */
    pub fn is_enclosed(&self, point: i32) -> bool {
        Self::positive_mod(point - self.start) <= Self::positive_mod(self.end - self.start)
    }

    /**
     * Checks if two sectors overlap.
     *
     * Returns `true` if the sectors share any angular range.
     * Used by SWAP* to prune route pairs that cannot benefit from exchanges.
     */
    pub fn overlap(sector1: &CircleSector, sector2: &CircleSector) -> bool {
        Self::positive_mod(sector2.start - sector1.start)
            <= Self::positive_mod(sector1.end - sector1.start)
            || Self::positive_mod(sector1.start - sector2.start)
                <= Self::positive_mod(sector2.end - sector2.start)
    }

    /**
     * Extends the sector to include a new point.
     *
     * If the point is not already enclosed, expands `start` or `end`
     * to include it, choosing the direction that minimizes angular span.
     */
    pub fn extend(&mut self, point: i32) {
        if !self.is_enclosed(point) {
            if Self::positive_mod(point - self.end) <= Self::positive_mod(self.start - point) {
                self.end = point;
            } else {
                self.start = point;
            }
        }
    }
}
