#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[derive(Copy, Clone)]
pub struct VoxelTree {
    pub base: *mut u32,
    allocator: *mut u32,
}

impl<'a> VoxelTree {
    pub fn allocate_node(mut self) -> u32 {
        let new = unsafe{ *self.allocator };
        unsafe{
                         *self.allocator += 10;
         }

        let node = self.get_node(new);
        node.childmask=0;
        node.flags=0;

        new
    }
    pub fn get_node(self, nptr: u32) -> &'a mut VoxelNode {
        unsafe {
            let mut ptr: *mut u32 = self.base;
            ptr = ptr.offset(nptr as isize);
            (ptr as *mut VoxelNode).as_mut().unwrap()
        }
    }
    pub fn allocate_and_get_node(&mut self) -> &'a mut VoxelNode {
        let ptr = self.allocate_node();
        self.get_node(ptr)
    }
}

#[derive(Copy, Clone)]
pub struct VoxelNode {
    pub childmask: u8,
    pub flags: u8,
    pub colour : [u8; 3],

    padding: [u8; 3],
    //8 to 64 more bytes of pointers... kinda
}

impl VoxelNode {

    pub fn put_child(&mut self, childindex: u8, childptr: u32) {
        let child: u8 = 1 << childindex;
        debug_assert_ne!(child, 0);
        unsafe {
            if (self.childmask & child) != 0 {
                *((&mut self.childmask as *mut u8 as *mut u32)
                    .offset(8 + (childindex as isize * 4))) = childptr;
            } else {
                let mut select: u8 = 1;
                let mut ptr: *mut u32 = &self.childmask as *const u8 as *mut u32;
                let mut moving = false;
                let mut holding = childptr;

                ptr = ptr.offset(2);
                loop {
                    if (select & child) != 0 {
                        moving = true;
                        self.childmask = self.childmask | child;
                    }

                    if (self.childmask & select) != 0 {
                        if moving {
                            let h = *ptr;
                            *ptr = holding;
                            holding = h;
                            ptr = ptr.offset(1);
                        } else {
                            ptr = ptr.offset(1);
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

    pub fn get_child(&self, tree: VoxelTree, childindex: u8) -> &mut VoxelNode {
        tree.get_node(self.get_child_ptr(childindex))
    }
    pub fn get_child_ptr(&self, childindex: u8) -> u32 {
        let child: u8 = 1 << childindex;
        debug_assert_ne!(child, 0);
        debug_assert_ne!(self.childmask, 0);
        debug_assert_ne!(self.childmask & child, 0);

        let mut select: u8 = 1;
        let mut ptr: *mut u32 = &self.childmask as *const u8 as *mut u32;
        unsafe {
            ptr = ptr.offset(2);
            loop {
                if (select & child) != 0 {
                    return *ptr;
                }

                if (self.childmask & select) != 0 {
                    ptr = ptr.offset(1);
                }
                select = select << 1;
            }
        }
    }
}
use antelope::cgmath::Vector3;
use num_traits::float::Float;

#[inline(always)]
pub fn oct_byte_to_vec<F>(byte: u8) -> Vector3<F>
where
    F: num_traits::float::Float,
{
    let y = F::from(byte_to_one(byte & 204)).unwrap();
    let x = F::from(byte_to_one(byte & 102)).unwrap();
    let z = F::from(byte_to_one(byte & 240)).unwrap();
    Vector3 {
        x: x - F::from(0.5).unwrap(),
        y: y - F::from(0.5).unwrap(),
        z: z - F::from(0.5).unwrap(),
    }
}

#[inline(always)]
pub fn byte_to_one(byte: u8) -> u8 {
    (1 as u8).saturating_sub((1 as u8).saturating_sub(byte))
}

pub fn allocate(size: usize) -> VoxelTree {
    unsafe {
        let base: *mut u32 =
            std::alloc::alloc(std::alloc::Layout::from_size_align(size, 4).unwrap())
            as *mut u32;

        let allocator = base as *mut u32;
        *allocator=1;

        VoxelTree {
            base,
            allocator,
        }
    }
}
