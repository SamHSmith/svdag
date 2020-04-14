#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

pub struct VoxelTree {
    base: *const u8,
}

pub struct VoxelNode {
    pub childmask: u8,
    pub flags: u8,

    padding: [u8; 6],
    //8 to 64 more bytes of pointers
}

impl VoxelNode {
    pub fn new(tree: &VoxelTree) -> *mut VoxelNode {
        return tree.base as *mut VoxelNode;
    }

    pub fn put_child(&mut self, childindex: u8, childptr: *const VoxelNode) {
        let child: u8 = 1 << childindex;
        debug_assert_ne!(child, 0);
        unsafe {
            if (self.childmask & child) != 0 {
                *self.get_child_ptr(childindex) = childptr;
            } else {
                let mut select: u8 = 1;
                let mut ptr: *mut *const VoxelNode =
                    &self.childmask as *const u8 as *mut *const VoxelNode;
                let mut moving = false;
                let mut holding = childptr;

                ptr = ptr.offset(8);
                loop {
                    if (select & child) != 0 {
                        moving = true;
                        self.childmask= self.childmask | child;
                    }

                    if (self.childmask & select) != 0 {
                        if moving {
                            let h = *ptr;
                            *ptr = holding;
                            holding = h;
                            ptr = ptr.offset(8);
                        } else {
                            ptr = ptr.offset(8);
                        }
                    }

                    if select == 128 {
                        return;
                    }
                    select = select << 1;
                }
            }
        }
    }

    pub fn get_child(&self, childindex: u8) -> *const VoxelNode {
        unsafe { *self.get_child_ptr(childindex) }
    }
    pub fn get_child_ptr(&self, childindex: u8) -> *mut *const VoxelNode {
        let child: u8 = 1 << childindex;
        debug_assert_ne!(child, 0);
        debug_assert_ne!(self.childmask, 0);
        debug_assert_ne!(self.childmask & child, 0);

        let mut select: u8 = 1;
        let mut ptr: *mut *const VoxelNode = &self.childmask as *const u8 as *mut *const VoxelNode;
        unsafe {
            ptr = ptr.offset(8);
            loop {
                if (select & child) != 0 {
                    return ptr;
                }

                if (self.childmask & select) != 0 {
                    ptr = ptr.offset(8);
                }
                select = select << 1;
            }
        }
    }
}
#[inline(always)]
pub fn oct_byte_to_vec<F>(byte: u8) -> (F, F, F) where F : num_traits::float::Float{
    let y = F::from(byte_to_one(byte & 204)).unwrap();
    let x = F::from(byte_to_one(byte & 102)).unwrap();
    let z = F::from(byte_to_one(byte & 240)).unwrap();
    (x, y, z)
}

#[inline(always)]
pub fn byte_to_one(byte: u8) -> u8{
    (1 as u8).saturating_sub((1 as u8).saturating_sub(byte))
}

pub fn allocate(size: usize) -> VoxelTree {
    unsafe {
        let base: *const u8 =
            std::alloc::alloc(std::alloc::Layout::from_size_align(size, 8).unwrap());

        VoxelTree { base: base }
    }
}
