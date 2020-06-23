use crate::VoxelNode;
use crate::VoxelTree;
use std::ops::IndexMut;

pub struct DenseVoxelData {
    pub width : usize,
    pub height : usize,
    pub depth : usize,
    pub data : Vec<VoxelNode>,
}

impl DenseVoxelData {// the dimensions are actually 2^n
    pub fn new(width : u32, height : u32, depth : u32) -> Self {
        DenseVoxelData {
            width: 2usize.pow(width),
            height: 2usize.pow(height),
            depth: 2usize.pow(depth),
            data : vec![VoxelNode::empty(); 2usize.pow(width) * 2usize.pow(height) * 2usize.pow(depth)]
        }
    }

    pub fn access(&mut self, x : usize, y : usize, z : usize) -> &mut VoxelNode {
        debug_assert!(x < self.width, "AAh, out of range.");
        debug_assert!(y < self.height, "AAh, out of range.");
        debug_assert!(z < self.depth, "AAh, out of range.");
        self.data.index_mut(x + (y * self.width) + (z * (self.width * self.height)))
    }

    pub fn to_sparse(&self) -> VoxelTree {
        let tree = allocate(self.width * self.height * self.depth * 3);
        
    }
}
