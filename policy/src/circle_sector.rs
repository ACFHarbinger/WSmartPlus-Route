#[derive(Clone, Copy, Default, Debug)]
pub struct CircleSector {
    pub start: i32,
    pub end: i32,
}

impl CircleSector {
    pub fn new(point: i32) -> Self {
        Self {
            start: point,
            end: point,
        }
    }

    // Positive modulo 65536
    fn positive_mod(i: i32) -> i32 {
        (i % 65536 + 65536) % 65536
    }

    pub fn is_enclosed(&self, point: i32) -> bool {
        Self::positive_mod(point - self.start) <= Self::positive_mod(self.end - self.start)
    }

    pub fn overlap(sector1: &CircleSector, sector2: &CircleSector) -> bool {
        Self::positive_mod(sector2.start - sector1.start)
            <= Self::positive_mod(sector1.end - sector1.start)
            || Self::positive_mod(sector1.start - sector2.start)
                <= Self::positive_mod(sector2.end - sector2.start)
    }

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
