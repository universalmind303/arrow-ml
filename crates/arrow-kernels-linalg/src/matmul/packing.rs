/// Pack A[ic..ic+mc, pc..pc+kc] into micro-panel layout.
///
/// Output layout: sequential micro-panels of `mr` rows.
/// Each micro-panel is stored as `[kc][mr]` — for each k-step,
/// `mr` consecutive values from the same column of the micro-panel.
/// Zero-pads if fewer than `mr` rows remain.
pub fn pack_a<T: Copy + Default>(
    a: &[T],
    lda: usize, // row stride of A (= k, the number of columns)
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    mr: usize,
    packed: &mut [T],
) {
    let mut offset = 0;
    let mut ir = 0;
    while ir < mc {
        let mr_actual = mr.min(mc - ir);
        if mr_actual == mr {
            // Fast path: full micro-panel — no zero-padding needed
            for p in 0..kc {
                let col = pc + p;
                for i in 0..mr {
                    // SAFETY: bounds are guaranteed by tiling — ic+ir+i < m, col < k
                    unsafe {
                        *packed.get_unchecked_mut(offset) =
                            *a.get_unchecked((ic + ir + i) * lda + col);
                    }
                    offset += 1;
                }
            }
        } else {
            // Edge case: partial micro-panel — zero-pad remaining rows
            for p in 0..kc {
                let col = pc + p;
                for i in 0..mr {
                    if i < mr_actual {
                        unsafe {
                            *packed.get_unchecked_mut(offset) =
                                *a.get_unchecked((ic + ir + i) * lda + col);
                        }
                    } else {
                        unsafe {
                            *packed.get_unchecked_mut(offset) = T::default();
                        }
                    }
                    offset += 1;
                }
            }
        }
        ir += mr;
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into micro-panel layout.
///
/// Output layout: sequential micro-panels of `nr` columns.
/// Each micro-panel is stored as `[kc][nr]` — for each k-step,
/// `nr` consecutive values from the same row of the micro-panel.
/// Zero-pads if fewer than `nr` columns remain.
pub fn pack_b<T: Copy + Default>(
    b: &[T],
    ldb: usize, // row stride of B (= n, the number of columns)
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    nr: usize,
    packed: &mut [T],
) {
    let mut offset = 0;
    let mut jr = 0;
    while jr < nc {
        let nr_actual = nr.min(nc - jr);
        if nr_actual == nr {
            // Fast path: full micro-panel — contiguous row copy
            for p in 0..kc {
                let row_start = (pc + p) * ldb + (jc + jr);
                // SAFETY: row_start + nr is within bounds due to tiling
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b.as_ptr().add(row_start),
                        packed.as_mut_ptr().add(offset),
                        nr,
                    );
                }
                offset += nr;
            }
        } else {
            // Edge case: partial micro-panel — copy valid + zero-pad
            for p in 0..kc {
                let row_start = (pc + p) * ldb + (jc + jr);
                for j in 0..nr {
                    if j < nr_actual {
                        unsafe {
                            *packed.get_unchecked_mut(offset) =
                                *b.get_unchecked(row_start + j);
                        }
                    } else {
                        unsafe {
                            *packed.get_unchecked_mut(offset) = T::default();
                        }
                    }
                    offset += 1;
                }
            }
        }
        jr += nr;
    }
}
