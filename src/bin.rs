use cgmath::prelude::One;
use cgmath::{Deg, Euler, InnerSpace, Matrix4, Quaternion, Vector3, Vector4};
use lib::render::cpu::CpuRenderer;
use lib::render::*;

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

const RPP: usize = 20;
const RBC: usize = 3;

use lib::*;

fn main() {
    use image::ImageBuffer;
    use image::Rgba;
    use std::sync::Arc;
    use vulkano::buffer::BufferUsage;
    use vulkano::buffer::CpuAccessibleBuffer;
    use vulkano::command_buffer::AutoCommandBufferBuilder;
    use vulkano::command_buffer::CommandBuffer;
    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
    use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
    use vulkano::device::Features;
    use vulkano::format::Format;
    use vulkano::image::Dimensions;
    use vulkano::image::StorageImage;
    use vulkano::instance::PhysicalDevice;
    use vulkano::pipeline::ComputePipeline;
    use vulkano::sync::GpuFuture;

    use glfw::{ClientApiHint, WindowHint};
    use vulkano::command_buffer::DynamicState;
    use vulkano::device::{Device, DeviceExtensions};
    use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
    use vulkano::image::SwapchainImage;
    use vulkano::pipeline::viewport::Viewport;
    use vulkano::pipeline::GraphicsPipeline;
    use vulkano::swapchain;
    use vulkano::swapchain::{
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
        SwapchainCreationError,
    };
    use vulkano::sync;
    use vulkano::sync::FlushError;

    use crossvulkan::*;

    let mut cv = init();

    let mut glfw = &mut cv.glfw;

    let instance = cv.vulkano_instance.clone();

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
    let (mut window, events) = glfw
        .create_window(512, 512, "Hello this is window", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window."); //TODO WRAP

    /// This method is called once during initialization, then again whenever the window is resized
    fn window_size_dependent_setup(
        images: &[Arc<SwapchainImage<()>>],
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        dynamic_state: &mut DynamicState,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        let dimensions = images[0].dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };
        dynamic_state.viewports = Some(vec![viewport]);

        images
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<dyn FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>()
    }

    let mut vksurf: vk_sys::SurfaceKHR = 0;
    assert_eq!(
        window.create_window_surface(cv.instance, std::ptr::null(), &mut vksurf),
        vk_sys::SUCCESS
    );

    let surface = Arc::new(unsafe {
        vulkano::swapchain::Surface::<()>::from_raw_surface(instance.clone(), vksurf, ())
    }); //TODO wrap this in a safe function

    let queue_family = physical
        .queue_families()
        .find(|&q| {
            q.supports_graphics()
                && q.supports_compute()
                && surface.is_supported(q).unwrap_or(false)
        })
        .expect("couldn't find a graphical queue family");

    let mut feat = Features::none();
    feat.shader_f3264 = true;
    feat.shader_storage_image_extended_formats = true;

    let mut exts = DeviceExtensions::none();
    exts.khr_storage_buffer_storage_class = true;
    exts.khr_swapchain = true;

    let (device, mut queues) = {
        Device::new(
            physical,
            &feat,
            &exts,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let mut tree = allocate(500);
    let node: &mut VoxelNode = tree.allocate_and_get_node();
    node.put_child(1, tree.allocate_node());
    node.get_child(tree, 1).flags = 1;
    node.get_child(tree, 1).colour = [20, 30, 90];
    node.get_child(tree, 1).emission = 255;
    node.get_child(tree, 1).roughness = 210;
    node.put_child(0, tree.allocate_node());
    let node2: &mut VoxelNode = node.get_child(tree, 0);
    node2.put_child(3, tree.allocate_node());
    node2.get_child(tree, 3).flags = 1;
    node2.get_child(tree, 3).colour = [130, 100, 2];
    node2.get_child(tree, 3).roughness = 35;
    node2.put_child(0, tree.allocate_node());
    node2.get_child(tree, 0).put_child(6, tree.allocate_node());
    node2.get_child(tree, 0).get_child(tree, 6).flags = 1;
    node2.get_child(tree, 0).get_child(tree, 6).colour = [198, 198, 200];
    node2.get_child(tree, 0).get_child(tree, 6).metalness = 255;
    node2.get_child(tree, 0).get_child(tree, 6).roughness = 150;
    node2.put_child(2, tree.allocate_node());
    node2.get_child(tree, 2).flags = 1;
    node2.get_child(tree, 2).colour = [255, 219, 145];
    node2.get_child(tree, 2).roughness = 10;
    node2.get_child(tree, 2).metalness = 255;
	node2.get_child(tree, 2).put_child(0, tree.allocate_node());
	node2.get_child(tree, 2).get_child(tree, 0).flags = 1;
	node2.get_child(tree, 2).get_child(tree, 0).colour = [255, 219, 145];
    node2.put_child(6, tree.allocate_node());
    node2.get_child(tree, 6).flags = 1;
    node2.get_child(tree, 6).colour = [50, 50, 50];
    node2.get_child(tree, 6).roughness = 249;
    node2.put_child(5, tree.allocate_node());
    node2.get_child(tree, 5).put_child(6, tree.allocate_node());
    node2.get_child(tree, 5).get_child(tree, 6).flags = 1;
    node2.get_child(tree, 5).get_child(tree, 6).colour = [255, 147, 41];
    node2.get_child(tree, 5).get_child(tree, 6).emission = 255;
    node2.get_child(tree, 5).get_child(tree, 6).roughness = 255;
    node2.put_child(7, tree.allocate_node());
    node2.get_child(tree, 7).flags = 1;
    node2.get_child(tree, 7).colour = [20, 50, 180];
    node2.get_child(tree, 7).roughness = 240;

    //node.flags =1;
    node.colour = [255, 183, 235];

    let mut campos = new_vec(-0.2, -0.75, -1.7);
    let mut camrot = new_vec(-10.0, 0.0, 0.0);
    let rotation = Quaternion::from(Euler {
        x: Deg(camrot.x),
        y: Deg(camrot.y),
        z: Deg(camrot.z),
    });

    let cubepos = new_vec(0.0, 0.0, 0.0);

    let mut cpurend = CpuRenderer::new(WIDTH, HEIGHT, 0, RBC, 1);
    cpurend.scenes[0] = VoxelScene {
        tree: tree,
        root: 1,
        camera_position: campos,
        camera_rotation: rotation,
    };

    use vulkano::buffer::CpuBufferPool;

    let cpubufferpool = CpuBufferPool::upload(device.clone());

    let vbuffer = unsafe {
        let mut bufusage = BufferUsage::none();
        bufusage.storage_buffer = true; //TODO potensial BufferUsage::storage_buffer()?
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            bufusage,
            false,
            (0..500)
                .into_iter()
                .map(|i| *(tree.base as *const VoxelNode as *const u32).offset(i)),
        )
        .unwrap()
    };

    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        // The alpha mode indicates how the alpha value of the final image will behave. For example
        // you can choose whether the window will be opaque or transparent.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // The dimensions of the window, only used to initially setup the swapchain.
        // NOTE:
        // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
        // swapchain size must use these dimensions.
        // These dimensions are always the same as the window dimensions
        //
        // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
        // These drivers will allow anything but the only sensible value is the window dimensions.
        //
        // Because for both of these cases, the swapchain needs to be the window dimensions, we just use that.
        let dimensions: [u32; 2] = [window.get_size().0 as u32, window.get_size().1 as u32];

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };

    use vulkano::format;
    use vulkano::image::AttachmentImage;
    use vulkano::image::ImageUsage;

    let mut _usage = ImageUsage::none();
    _usage.sampled = true;
    _usage.transfer_destination = true;

    let rayimages: Vec<Arc<AttachmentImage<format::R8G8B8A8Unorm>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            AttachmentImage::with_usage(
                device.clone(),
                [WIDTH as u32, HEIGHT as u32],
                format::R8G8B8A8Unorm,
                _usage,
            )
            .unwrap()
        })
        .collect();

    let imagestep: Vec<Arc<StorageImage<format::R16G16B16A16Unorm>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            StorageImage::new(
                device.clone(),
                Dimensions::Dim3d {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                    depth: RPP as u32,
                },
                vulkano::format::R16G16B16A16Unorm,
                Some(queue.family()),
            )
            .unwrap()
        })
        .collect();

    let imagestep2: Vec<Arc<StorageImage<format::R16G16B16A16Unorm>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            StorageImage::new(
                device.clone(),
                Dimensions::Dim2d {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                },
                vulkano::format::R16G16B16A16Unorm,
                Some(queue.family()),
            )
            .unwrap()
        })
        .collect();

    let cpuimages: Vec<Arc<StorageImage<format::R16G16B16A16Unorm>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            StorageImage::new(
                device.clone(),
                Dimensions::Dim2d {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                },
                vulkano::format::R16G16B16A16Unorm,
                Some(queue.family()),
            )
            .unwrap()
        })
        .collect();

    let imagecombines: Vec<Arc<StorageImage<Format>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            StorageImage::new(
                device.clone(),
                Dimensions::Dim2d {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                },
                Format::R16G16B16A16Unorm,
                Some(queue.family()),
            )
            .unwrap()
        })
        .collect();

    let outimages: Vec<Arc<StorageImage<Format>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            StorageImage::new(
                device.clone(),
                Dimensions::Dim2d {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                },
                Format::R8G8B8A8Unorm,
                Some(queue.family()),
            )
            .unwrap()
        })
        .collect();

    // We now create a buffer that will store the shape of our triangle.
    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 0.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
                Vertex {
                    position: [0.0, 1.0],
                },
                Vertex {
                    position: [0.0, 0.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450
				layout(location = 0) in vec2 position;
        layout(location = 0) out vec2 fragposition;
				void main() {
					gl_Position = vec4((position * 2.0) - vec2(1.0), 0.0, 1.0);
          fragposition = position;
				}
			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450
        layout(location = 0) in vec2 position;
				layout(location = 0) out vec4 f_color;

        layout(set = 0, binding = 0) uniform sampler2D img;
				void main() {
					f_color = texture(img, position);
				}
			"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // same format as the swapchain.
                    format: swapchain.format(),
                    // TODO:
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            // We need to indicate the layout of the vertices.
            // The type `SingleBufferDefinition` actually contains a template parameter corresponding
            // to the type of each vertex. But in this code it is automatically inferred.
            .vertex_input_single_buffer()
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one. The `main` word of `main_entry_point` actually corresponds to the name of
            // the entry point.
            .vertex_shader(vs.main_entry_point(), ())
            // The content of the vertex buffer describes a list of triangles.
            .triangle_list()
            // Use a resizable viewport set to draw over the entire window
            .viewports_dynamic_scissors_irrelevant(1)
            // See `vertex_shader`.
            .fragment_shader(fs.main_entry_point(), ())
            // We have to indicate which subpass of which render pass this pipeline is going to be used
            // in. The pipeline will only be usable from this particular subpass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define VBUFFER_SIZE 500

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16) uniform writeonly image3D img;
layout(set = 0, binding = 1) buffer VoxelBuffer {
    vec4[] data;
};

layout(push_constant) uniform pushConstants {
    vec3 campos;
    vec3 right;
    vec3 up;
    vec3 forward;
    uint framenum;
} pc;

float rand(vec2 co){
    return fract(sin(dot(vec2(co.x + float(pc.framenum * 5), co.y * float(pc.framenum + 2)),vec2(12.9898,78.233))) * 43758.5453);
}

vec3 reflectvec(vec3 v, vec3 normal){
    return v + (-2.0 * dot(v, normal) * normal);
}

const float F0 = 0.04;
float fresnel(vec3 v, vec3 normal){
    return F0 + ((1.0 - F0) * pow(1.0 - dot(normal,v),5.0));
}

vec3 rand_dir_from_surf(vec3 normal, vec2 co){
    vec3 v = vec3(rand(co), rand(co * 2), rand(co *3));

    if(dot(v, normal) < 0.0) {
        v += (-2.0 * dot(v, normal) * normal);
    }
    normalize(v);
    return v;
}

vec3 oct_byte_to_vec(uint byte){ // 204 102 240
    return vec3(float(bool(byte & 102)), float(bool(byte & 204)), float(bool(byte & 240))) - vec3(0.5);
}

vec3 biggest_axis(vec3 v){
    float x = float(uint(abs(v.x) > abs(v.y) && abs(v.x) > abs(v.z)));
    float y = float(uint(abs(v.y) > abs(v.x) && abs(v.y) > abs(v.z)));
    float z = float(uint(abs(v.z) > abs(v.x) && abs(v.z) > abs(v.y)));

    return v * vec3(x,y,z);
}

uint get_byte(uint src, uint byte){
    src = src << (3 - byte) * 8;
    src = src >> 3 * 8;
    return src;
}

uint read_vbuffer(uint address){
    uint vecadr = uint((address - mod(address, 4)) / 4);
    vec4 vec = data[vecadr];

    return floatBitsToUint(vec[uint(mod(address, 4))]);
}

const float EPSILON = 0.0000002;


bool inbox(vec3 pos, float size){
    return (pos.x + EPSILON >= size * -0.5) && (pos.x - EPSILON <= size * 0.5) && (pos.y + EPSILON >= size * -0.5) && (pos.y - EPSILON <= size * 0.5) && (pos.z + EPSILON >= size * -0.5) && (pos.z - EPSILON <= size * 0.5);
}

struct RayTarget{
    vec3 hitlocation;
    vec3 nodelocation;
    vec3 hitnormal;
    vec3 raydir;
    float t;
    float specular;
    uint node;
};

RayTarget cast_ray_v_box(vec3 start, vec3 dir, float size, vec3 nodelocation, uint node, vec3 globalray, float specular) {
    float t1 = (size * -0.5 - start.x) / dir.x;
    float t2 = (size * 0.5 - start.x) / dir.x;

    float t3 = (size * -0.5 - start.y) / dir.y;
    float t4 = (size * 0.5 - start.y) / dir.y;

    float t5 = (size * -0.5 - start.z) / dir.z;
    float t6 = (size * 0.5 - start.z) / dir.z;

    vec3 pos1 = start + dir * t1;
    vec3 pos2 = start + dir * t2;
    vec3 pos3 = start + dir * t3;
    vec3 pos4 = start + dir * t4;
    vec3 pos5 = start + dir * t5;
    vec3 pos6 = start + dir * t6;

    float b1 = (1.0 - float(inbox(pos1, size) && t1 > 0)) * FLT_MAX;
    float b2 = (1.0 - float(inbox(pos2, size) && t2 > 0)) * FLT_MAX;
    float b3 = (1.0 - float(inbox(pos3, size) && t3 > 0)) * FLT_MAX;
    float b4 = (1.0 - float(inbox(pos4, size) && t4 > 0)) * FLT_MAX;
    float b5 = (1.0 - float(inbox(pos5, size) && t5 > 0)) * FLT_MAX;
    float b6 = (1.0 - float(inbox(pos6, size) && t6 > 0)) * FLT_MAX;

    b1 = max(b1, t1);
    b2 = max(b2, t2);
    b3 = max(b3, t3);
    b4 = max(b4, t4);
    b5 = max(b5, t5);
    b6 = max(b6, t6);

    float min = min(b1, min(b2, min(b3, min(b4, min(b5, b6)))));

    if(abs(b1 - min) < EPSILON){return RayTarget(nodelocation + pos1, nodelocation, normalize(biggest_axis(pos1)), globalray, b1, specular, node);}
    if(abs(b2 - min) < EPSILON){return RayTarget(nodelocation + pos2, nodelocation, normalize(biggest_axis(pos2)), globalray, b2, specular, node);}
    if(abs(b3 - min) < EPSILON){return RayTarget(nodelocation + pos3, nodelocation, normalize(biggest_axis(pos3)), globalray, b3, specular, node);}
    if(abs(b4 - min) < EPSILON){return RayTarget(nodelocation + pos4, nodelocation, normalize(biggest_axis(pos4)), globalray, b4, specular, node);}
    if(abs(b5 - min) < EPSILON){return RayTarget(nodelocation + pos5, nodelocation, normalize(biggest_axis(pos5)), globalray, b5, specular, node);}
    if(abs(b6 - min) < EPSILON){return RayTarget(nodelocation + pos6, nodelocation, normalize(biggest_axis(pos6)), globalray, b6, specular, node);}
}
/*
struct VoxelNode{
    childmask: u8,
    flags: u8,
    colour : [u8; 3],

    padding: [u8; 3],
    //8 to 64 more bytes of pointers... kinda 32 do
}
*/

uint get_voxel_childmask(uint ptr){
    return get_byte(read_vbuffer(ptr),0);
}
uint get_voxel_flags(uint ptr){
    return get_byte(read_vbuffer(ptr),1);
}
vec3 get_voxel_colour(uint ptr){
    return vec3(get_byte(read_vbuffer(ptr),2), get_byte(read_vbuffer(ptr),3) , get_byte(read_vbuffer(ptr + 1),0)) / 255.0;
}
float get_voxel_emission(uint ptr){
    return float(get_byte(read_vbuffer(ptr + 1),1) / 255.0);
}
float get_voxel_metalness(uint ptr){
    return float(get_byte(read_vbuffer(ptr + 1),2) / 255.0);
}
float get_voxel_roughness(uint ptr){
    return float(get_byte(read_vbuffer(ptr + 1),3)) / 255.0;
}
uint get_voxel_child(uint voxelptr, uint childindex){
    uint child = 1 << childindex;

    uint select = 1;
    uint ptr = voxelptr + 2;
    while (true) {
        if((select & child) != 0) {
            return read_vbuffer(ptr);
        }

        if((get_voxel_childmask(voxelptr) & select) != 0) {
            ptr++;
        }
        select = select << 1;
    }
}

const uint stacksize = 16;

struct StackFrame {
    vec3 location;
    float size;
    uint node;
    uint subnode;
};

RayTarget cast_ray_voxel(vec3 start, vec3 dir, uint root, float specular) {
    float size = 1.0;

    RayTarget closestleaf = RayTarget(vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),0.0,0.0,0);
    closestleaf.t = FLT_MAX;

    StackFrame[stacksize] frames;
    int stackindex = -1;

    RayTarget h = cast_ray_v_box(start - vec3(0.0), dir, size, vec3(0.0), root, dir, specular);

    if(h.t < FLT_MAX && h.t > 0){
        stackindex++;
        frames[stackindex] = StackFrame(vec3(0.0), size, root, 0);
	}
	
    while(stackindex >= 0) {
		int i = stackindex;
        uint child = 1 << frames[i].subnode;
        if((get_voxel_childmask(frames[i].node) & child) != 0) {
            vec3 loc = frames[i].location + (oct_byte_to_vec(child) * frames[i].size / 2.0);
            RayTarget hit = cast_ray_v_box(start - loc, dir, frames[i].size / 2.0, loc, get_voxel_child(frames[i].node, frames[i].subnode), dir, specular);

            if(hit.t > 0 && hit.t < FLT_MAX){
                if((get_voxel_flags(hit.node) & 1) != 0 && hit.t > 0 && hit.t < closestleaf.t){
                    closestleaf = hit;
                }
                if((get_voxel_childmask(hit.node) & child) != 0){
					stackindex++;
                    frames[stackindex] = StackFrame(loc, frames[i].size / 2.0, hit.node, 0);
                }
            }
        }
		frames[i].subnode++;
		
		if(stackindex >= stacksize){
			stackindex--;
		}
		
		if(frames[stackindex].subnode > 7){
			stackindex--;
		}
    }
    return closestleaf;
}

const uint RBC = 4; // + 1
const float EMISSION = 15.0;

vec3 calculate_colour(RayTarget t, vec3 nc, float ns){
    float specular = ns * ((fresnel(-t.raydir, t.hitnormal) * (1.0 - get_voxel_roughness(t.node))) * ((0.5 * (1.0 - get_voxel_metalness(t.node))) + (0.98 * get_voxel_metalness(t.node))));
    vec3 specular_highlight = ((vec3(0.5) * (1.0 - get_voxel_metalness(t.node)))
            + (get_voxel_colour(t.node) * get_voxel_metalness(t.node)))
            * specular;

    vec3 colour = vec3(0.0);

    colour += specular_highlight * nc;
    colour += (1.0 - ns) * 0.5 * (1.0 - get_voxel_metalness(t.node)) * get_voxel_colour(t.node) * nc;

    colour *= (1.0 - get_voxel_emission(t.node));
    colour += get_voxel_colour(t.node) * get_voxel_emission(t.node) * EMISSION;
    colour *= length(t.hitnormal);

    return colour;
}

void main() {
    vec2 coords = vec2(gl_GlobalInvocationID.xy) / vec2(imageSize(img).xy);
    vec2 pixelsize = vec2(1.0) / vec2(imageSize(img));

    vec3 colour = vec3(0.0);

    RayTarget groups[RBC];


        vec2 randcoord = coords + (pixelsize * vec2(rand(vec2(coords) * float(gl_GlobalInvocationID.z)),
 rand(vec2(coords) * float(gl_GlobalInvocationID.z + 1))));
        vec3 raydir = ((randcoord.x * 2.0 - 1.0) * vec3(pc.right)) + ((randcoord.y * 2.0 - 1.0) * vec3(pc.up)) + vec3(pc.forward);
        vec3 start =  vec3(pc.campos);

        groups[0] = cast_ray_voxel(start, normalize(raydir), 1, 0);

        if(length(groups[0].hitnormal) < EPSILON){
imageStore(img, ivec3(gl_GlobalInvocationID.xyz), vec4(0.0,0.0,0.0,1.0));
return; }

        for(uint g = 1; g < RBC; g++){
            float roughness = get_voxel_roughness(groups[g-1].node);
            float specular= round(float(rand(randcoord * (g + 1)) > (0.5 * (1.0 - get_voxel_metalness(groups[g-1].node)))));
            vec3 newdir=rand_dir_from_surf(groups[g-1].hitnormal, vec2(randcoord) * float(g));
            vec3 specdir=(roughness * newdir) + ((1.0 - roughness) * reflectvec(groups[g-1].raydir, groups[g-1].hitnormal));
            groups[g] = cast_ray_voxel(groups[g-1].hitlocation + ((EPSILON * 2) * groups[g-1].hitnormal), normalize((newdir * (1.0-specular)) + (specdir * specular)), 1, specular);
        }
        vec3[RBC] colours;
        colours[RBC - 1]= calculate_colour(groups[RBC-1], vec3(0.0), 0.0);


        for(int g = int(RBC) - 2; g >= 0; g--){
            colours[g]=calculate_colour(groups[g], colours[g + 1], groups[g + 1].specular);
        }

        colour = colours[0];

    float biggest= 0.0;
    biggest = max(biggest, colour.x);
    biggest = max(biggest, colour.y);
    biggest = max(biggest, colour.z);


    colour /= biggest;

    imageStore(img, ivec3(gl_GlobalInvocationID.xyz), vec4(colour, min(max(biggest / EMISSION, 0.0), 1.0)));
}"
        }
    }

    mod cs2 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define VBUFFER_SIZE 500

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16) uniform readonly image3D img;
layout(set = 0, binding = 1, rgba16) uniform writeonly image2D img2;

const float EMISSION = 15.0;

void main() {
    vec3 final = vec3(0.0);

    for(uint i = 0; i < imageSize(img).z; i++){
        vec4 colour = imageLoad(img, ivec3(gl_GlobalInvocationID.xy, i));
        final += colour.xyz * (colour.w * EMISSION);
    }

    final /= imageSize(img).z;

    float alpha = max(final.x, max(final.y, final.z));

    final /= alpha;

    imageStore(img2, ivec2(gl_GlobalInvocationID.xy), vec4(final, alpha / EMISSION));
}
"
        }
    }

    mod cs3 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define VBUFFER_SIZE 500

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16) uniform readonly image2D imgin;
layout(set = 1, binding = 0, rgba16) uniform image2D imgout;

layout(push_constant) uniform pushConstants {
    uint rpp;
    uint so_far_rpp;
} pc;

void main() {
    vec4 final = imageLoad(imgin, ivec2(gl_GlobalInvocationID.xy)) * (float(pc.rpp) / float(pc.rpp + pc.so_far_rpp));

    final += imageLoad(imgout, ivec2(gl_GlobalInvocationID.xy)) * (float(pc.so_far_rpp) / float(pc.rpp + pc.so_far_rpp));

    imageStore(imgout, ivec2(gl_GlobalInvocationID.xy), final);
    //imageStore(imgout, ivec2(gl_GlobalInvocationID.xy), vec4(1.0));
}
"
        }
    }

    mod cs4 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define VBUFFER_SIZE 500

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16) uniform readonly image2D imgin;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D imgout;


vec3 toSRGB(vec3 linear)
{
    bvec3 cutoff = lessThan(linear, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(linear, vec3(1.0) / vec3(2.4)) - vec3(0.055);
    vec3 lower = vec3(12.92) * linear;

    return mix(higher, lower, cutoff);
}

const float EMISSION = 15.0;

void main() {
    vec4 final = imageLoad(imgin, ivec2(gl_GlobalInvocationID.xy));

    imageStore(imgout, ivec2(gl_GlobalInvocationID.xy), vec4(toSRGB(final.xyz * final.w * EMISSION), 1.0));
}
"
        }
    }

    let shader = cs::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let shader2 = cs2::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline2 = Arc::new(
        ComputePipeline::new(device.clone(), &shader2.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let shader3 = cs3::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline3 = Arc::new(
        ComputePipeline::new(device.clone(), &shader3.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let shader4 = cs4::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline4 = Arc::new(
        ComputePipeline::new(device.clone(), &shader4.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    use vulkano::descriptor::descriptor_set::DescriptorSet;

    let mut sets = Vec::new();
    for i in 0..images.len() {
        sets.push(Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline
                    .layout()
                    .descriptor_set_layout(0)
                    .unwrap()
                    .clone(),
            )
            .add_image(imagestep[i].clone())
            .unwrap()
            .add_buffer(vbuffer.clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }

    let mut set2 = Vec::new();
    for i in 0..images.len() {
        set2.push(Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline2
                    .layout()
                    .descriptor_set_layout(0)
                    .unwrap()
                    .clone(),
            )
            .add_image(imagestep[i].clone())
            .unwrap()
            .add_image(imagestep2[i].clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }

    use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;

    let mut set3pool = FixedSizeDescriptorSetsPool::new(
        compute_pipeline3
            .layout()
            .descriptor_set_layout(0)
            .unwrap()
            .clone(),
    );

    let mut set3 = Vec::new();
    for i in 0..images.len() {
        set3.push(Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline3
                    .layout()
                    .descriptor_set_layout(1)
                    .unwrap()
                    .clone(),
            )
            .add_image(imagecombines[i].clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }

    let mut set4 = Vec::new();
    for i in 0..images.len() {
        set4.push(Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline4
                    .layout()
                    .descriptor_set_layout(0)
                    .unwrap()
                    .clone(),
            )
            .add_image(imagecombines[i].clone())
            .unwrap()
            .add_image(outimages[i].clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }
    use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
    let _sampler = Sampler::new(
        device.clone(),
        Filter::Nearest,
        Filter::Nearest,
        MipmapMode::Nearest,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();

    let mut setg = Vec::new();
    for i in 0..images.len() {
        setg.push(Arc::new(
            PersistentDescriptorSet::start(
                pipeline.layout().descriptor_set_layout(0).unwrap().clone(),
            )
            .add_sampled_image(rayimages[i].clone(), _sampler.clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }

    let right = rotation
        * Vector3::<f64> {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
    let up = rotation
        * Vector3::<f64> {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
    let forward = rotation
        * Vector3::<f64> {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };

    fn gen_push_const(
        campos: Vector3<f64>,
        right: Vector3<f64>,
        up: Vector3<f64>,
        forward: Vector3<f64>,
        framenum: u32,
    ) -> cs::ty::pushConstants {
        let push_constants = cs::ty::pushConstants {
            campos: [campos.x as f32, campos.y as f32, campos.z as f32],
            right: [right.x as f32, right.y as f32, right.z as f32],
            up: [up.x as f32, up.y as f32, up.z as f32],
            forward: [forward.x as f32, forward.y as f32, forward.z as f32],
            _dummy0: [0; 4],
            _dummy1: [0; 4],
            _dummy2: [0; 4],
            framenum: framenum,
        };
        return push_constants;
    }
    {
        let buf = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            (0..WIDTH * HEIGHT * 4).map(|_| 0u8),
        )
        .expect("failed to create buffer");

        let cpubuffer = cpurend.finish_render(
            instance.clone(),
            device.clone(),
            queue.clone(),
            cpubufferpool.clone(),
            cpuimages[0].clone(),
            0,
        );

        let tempset = set3pool
            .next()
            .add_image(cpuimages[0].clone())
            .unwrap()
            .build()
            .unwrap();

        let gpubuffer = AutoCommandBufferBuilder::secondary_compute_one_time_submit(
            device.clone(),
            queue.family(),
        )
        .unwrap()
        .dispatch(
            [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, RPP as u32],
            compute_pipeline.clone(),
            (sets[0].clone()),
            gen_push_const(campos, right, up, forward, 0u32),
        )
        .unwrap()
        .dispatch(
            [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
            compute_pipeline2.clone(),
            (set2[0].clone()),
            (),
        )
        .unwrap()
        .build()
        .unwrap();

        let command_buffer = unsafe {
            AutoCommandBufferBuilder::new(device.clone(), queue.family())
                .unwrap()
                .execute_commands(cpubuffer)
                .unwrap()
                .execute_commands(gpubuffer)
                .unwrap()
                .dispatch(
                    [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                    compute_pipeline3.clone(),
                    (tempset, set3[0].clone()),
                    (1 as u32, 0 as u32),
                )
                .unwrap()
                .dispatch(
                    [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                    compute_pipeline4.clone(),
                    (set4[0].clone()),
                    (),
                )
                .unwrap()
                .copy_image_to_buffer(outimages[0].clone(), buf.clone())
                .unwrap()
                .build()
                .unwrap()
        };

        let gpustart = std::time::Instant::now();

        let finished = command_buffer.execute(queue.clone()).unwrap();
        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let gpuend = std::time::Instant::now();

        let buffer_content = buf.read().unwrap();
        let image =
            ImageBuffer::<Rgba<u8>, _>::from_raw(WIDTH as u32, HEIGHT as u32, &buffer_content[..])
                .unwrap();
        image.save("image2.png").unwrap();

        println!("Cpu took {} ms", (gpuend - gpustart).as_millis());
    }

    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    use std::time::Instant;

    let mut framenum: u32 = 0;
    let mut framecounter: u32 = 0;
    let mut lastframetimeprintout = Instant::now();

    while !window.should_close() {
        glfw.poll_events();

        framenum += 1;

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if recreate_swapchain {
            // Get the new dimensions of the window.
            let dimensions: [u32; 2] = [window.get_size().0 as u32, window.get_size().1 as u32];

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            swapchain = new_swapchain;
            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);
            recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
        // will still work, but it may not display correctly. With some drivers this can be when
        // the window resizes, but it may not cause the swapchain to become out of date.
        if suboptimal {
            recreate_swapchain = true;
        }

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

        campos += forward / 1000.0;

        cpurend.scenes[0] = VoxelScene {
            tree: tree,
            root: 1,
            camera_position: campos,
            camera_rotation: rotation,
        };
/*
        let cpubuffer = cpurend.finish_render(
            instance.clone(),
            device.clone(),
            queue.clone(),
            cpubufferpool.clone(),
            cpuimages[image_num].clone(),
            0,
        );
*/
        let gpubuffer = AutoCommandBufferBuilder::secondary_compute_one_time_submit(
            device.clone(),
            queue.family(),
        )
        .unwrap()
        .dispatch(
            [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, RPP as u32],
            compute_pipeline.clone(),
            (sets[image_num].clone()),
            gen_push_const(campos, right, up, forward, framenum),
        )
        .unwrap()
        .dispatch(
            [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
            compute_pipeline2.clone(),
            (set2[image_num].clone()),
            (),
        )
        .unwrap()
        .build()
        .unwrap();
        let mut tempsets = Vec::new();
		/*
        tempsets.push(
            set3pool
                .next()
                .add_image(cpuimages[image_num].clone())
                .unwrap()
                .build()
                .unwrap(),
        );*/
        tempsets.push(
            set3pool
                .next()
                .add_image(imagestep2[image_num].clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let mut rppsets = Vec::new();
        //rppsets.push(cpurend.rpp);
        rppsets.push(RPP);
        let mut sofarRPP = 0;

        let mut command_buffer_build = unsafe {
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .clear_color_image(
                    cpuimages[image_num].clone(),
                    vulkano::format::ClearValue::Float([0.0; 4]),
                )
                .unwrap()
                .clear_color_image(
                    imagestep2[image_num].clone(),
                    vulkano::format::ClearValue::Float([0.0; 4]),
                )
                //.execute_commands(cpubuffer)
                //.unwrap()
                .unwrap()
                .execute_commands(gpubuffer)
                .unwrap()
        };
        for i in 0..1 {
            let rpp = rppsets.pop().unwrap();
        command_buffer_build = command_buffer_build
            .dispatch(
                [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                compute_pipeline3.clone(),
                (tempsets.pop().unwrap(), set3[image_num].clone()),
                (rpp as u32, sofarRPP as u32),
            )
                .unwrap();
            sofarRPP += rpp;
}
        let command_buffer = command_buffer_build
            .dispatch(
                [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                compute_pipeline4.clone(),
                (set4[image_num].clone()),
                (),
            )
            .unwrap()
            .copy_image(
                outimages[image_num].clone(),
                [0; 3],
                0,
                0,
                rayimages[image_num].clone(),
                [0; 3],
                0,
                0,
                [WIDTH as u32, HEIGHT as u32, 1],
                1,
            )
            .unwrap()
            // Before we can draw, we have to *enter a render pass*. There are two methods to do
            // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
            // not covered here.
            //
            // The third parameter builds the list of values to clear the attachments with. The API
            // is similar to the list of attachments when building the framebuffers, except that
            // only the attachments that use `load: Clear` appear in the list.
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .draw(
                pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
                (setg[image_num].clone()),
                (),
            )
            .unwrap()
            // We leave the render pass by calling `draw_end`. Note that if we had multiple
            // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
            // next subpass.
            .end_render_pass()
            .unwrap()
            // Finish building the command buffer by calling `build`.
            .build()
            .unwrap();

        let future = previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            // The color output is now expected to contain our triangle. But in order to show it on
            // the screen, we have to *present* the image by calling `present`.
            //
            // This function does not actually present the image immediately. Instead it submits a
            // present command at the end of the queue. This means that it will only be presented once
            // the GPU has finished executing the command buffer that draws the triangle.
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
        }
        framecounter += 1;
        if (Instant::now() - lastframetimeprintout).as_secs() > 5 {
            println!(
                "Avg frame time: {}",
                (Instant::now() - lastframetimeprintout).as_millis() as f64 / framecounter as f64
            );
            framecounter = 0;
            lastframetimeprintout = Instant::now();
        }
    }
}
