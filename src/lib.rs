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

unsafe impl Send for VoxelTree {
    
}

unsafe impl Sync for VoxelTree {
    
}

#[derive(Copy, Clone)]
pub struct VoxelNode {
    pub childmask: u8,
    pub flags: u8,
    pub colour : [u8; 3],
    pub emission : u8,
    pub metalness : u8,
    pub roughness : u8,
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
use antelope::cgmath::{Vector3, InnerSpace};
use num_traits::float::Float;

#[inline(always)]
pub fn random_mix(a: f64, b: f64) -> f64 {
    let mix: f64 = rand::random();
    return (mix * a) + ((1.0 - mix) * b);
}
#[inline(always)]
pub fn new_vec(x: f64, y: f64, z: f64) -> Vector3<f64> {
    return Vector3 { x, y, z };
}


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
#[inline(always)]
pub fn get_biggest_axis(v :Vector3<f64>) -> Vector3<f64> {
    let x = (v.x.abs() > v.y.abs() && v.x.abs() > v.z.abs()) as i32 as f64;
    let y = (v.y.abs() > v.x.abs() && v.y.abs() > v.z.abs()) as i32 as f64;
    let z = (v.z.abs() > v.x.abs() && v.z.abs() > v.y.abs()) as i32 as f64;

    Vector3{x:x*v.x,y:y*v.y,z:z*v.z}
}

#[inline(always)]
pub fn reflect(v : Vector3<f64>, normal : Vector3<f64>) -> Vector3<f64> {
    v + (-2.0 * v.dot(normal) * normal)
}

#[inline(always)]
pub fn fresnel(v : Vector3<f64>, normal : Vector3<f64>) ->f64 {
    let F0 = 0.04;
    F0 + ((1.0 - F0) * (1.0 - normal.dot(v)).powf(5.0))
}

#[inline(always)]
pub fn point_wise_mul(v : Vector3<f64>, v2 : Vector3<f64>) -> Vector3<f64> {
    Vector3::<f64> {x:v.x*v2.x, y:v.y*v2.y, z: v.z*v2.z}
}

pub fn rand_dir_from_surf(normal : Vector3<f64>) -> Vector3<f64> {
    let mut v = Vector3::<f64> {x: rand::random(), y: rand::random(), z: rand::random(),};

    if v.dot(normal) < 0.0 {
        v += (-2.0 * v.dot(normal) * normal);
    }
    v.normalize()
}

pub fn linear_to_srgb(colour: Vector3<f64>) -> [u8; 3] {
    [lin_to_srgb(colour.x), lin_to_srgb(colour.y), lin_to_srgb (colour.z)]
}

fn lin_to_srgb(c: f64) -> u8 {
    let mut varR=c.min(1.0);
    if (varR > 0.0031308) {
        varR = 1.055 * (varR.powf(1.0 / 2.4)) - 0.055;
    } else {
        varR = 12.92 * varR;
    }
    return (varR * 255.0) as u8;
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
