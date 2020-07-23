use cgmath::prelude::One;
use cgmath::{Deg, Euler, InnerSpace, Matrix4, Quaternion, Vector3, Vector4};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

const RPP_mult: u32 = 1;
const RPP_buffer: usize = 8;

const SNAP_RPP_mult: u32 = 180;
const SNAP_LENGTH: usize = 100;

use svdag::*;

fn main() {
    use chrono::prelude::*;

    let mut snapcount = 1usize;
    let mut snaptime = Utc::now();

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

    use crossvulkan::CrossWindow;
    let mut window = CrossWindow::new_windowed(512, 512, &mut cv);

    /// This method is called once during initialization, then again whenever the window.window is resized
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
        window
            .window
            .create_window_surface(cv.instance, std::ptr::null(), &mut vksurf),
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
    node.get_child(tree, 1).colour = [70, 50, 90];
    node.get_child(tree, 1).emission = 0;
    node.get_child(tree, 1).roughness = 210;
    node.put_child(0, tree.allocate_node());
    let node2: &mut VoxelNode = node.get_child(tree, 0);
    node2.put_child(2, tree.allocate_node());
    node2.get_child(tree, 2).flags = 1;
    node2.get_child(tree, 2).colour = [130, 100, 2];
    node2.get_child(tree, 2).roughness = 35;
    node2.put_child(0, tree.allocate_node());
    node2.get_child(tree, 0).put_child(6, tree.allocate_node());
    node2.get_child(tree, 0).get_child(tree, 6).flags = 1;
    node2.get_child(tree, 0).get_child(tree, 6).colour = [198, 198, 200];
    node2.get_child(tree, 0).get_child(tree, 6).metalness = 255;
    node2.get_child(tree, 0).get_child(tree, 6).roughness = 150;
    node2.put_child(3, tree.allocate_node());
    node2.get_child(tree, 3).flags = 1;
    node2.get_child(tree, 3).colour = [255, 219, 145];
    node2.get_child(tree, 3).roughness = 10;
    node2.get_child(tree, 3).metalness = 255;
    node2.put_child(6, tree.allocate_node());
    node2.get_child(tree, 6).flags = 1;
    node2.get_child(tree, 6).colour = [50, 50, 50];
    node2.get_child(tree, 6).roughness = 249;
    node2.put_child(5, tree.allocate_node());
    node2.get_child(tree, 5).put_child(3, tree.allocate_node());
    node2.get_child(tree, 5).get_child(tree, 3).flags = 1;
    node2.get_child(tree, 5).get_child(tree, 3).colour = [255, 147, 41];
    node2.get_child(tree, 5).get_child(tree, 3).emission = 55;
    node2.get_child(tree, 5).get_child(tree, 3).roughness = 255;
    node2.put_child(7, tree.allocate_node());
    node2.get_child(tree, 7).flags = 1;
    node2.get_child(tree, 7).colour = [20, 50, 180];
    node2.get_child(tree, 7).roughness = 240;

    //test code
    use crate::dense::*;
    let mut dense = DenseVoxelData::new(3);
    dense.access_mut(0, 5, 7).flags = 1;
    dense.access_mut(0, 5, 7).colour = [20, 70, 200];
    dense.access_mut(0, 5, 7).emission = 112;
    dense.access_mut(0, 5, 7).roughness = 150;
    dense.access_mut(7, 6, 0).flags = 1;
    dense.access_mut(7, 6, 0).colour = [250, 100, 100];
    dense.access_mut(7, 6, 0).emission = 42;
    dense.access_mut(7, 6, 0).roughness = 150;

    for x in 0..8 {
        for z in 0..8 {
            dense.access_mut(x, 7, z).flags = 1;
            dense.access_mut(x, 7, z).emission = 0;
            dense.access_mut(x, 7, z).roughness = 230;
            dense.access_mut(x, 7, z).colour = [200, 200, 200];
        }
    }
    for y in 2..8 {
        for x in 3..4 {
            dense.access_mut(3, y, x).flags = 1;
            dense.access_mut(3, y, x).colour = [255, 219, 145];
            dense.access_mut(3, y, x).roughness = 10;
            dense.access_mut(3, y, x).emission = 0;
            dense.access_mut(3, y, x).metalness = 255;
        }
    }

    let treeother = dense.to_sparse();

    /*let mut tree = allocate(1000);
    let rootid = tree.allocate_node();
    let broot = tree.get_node(rootid);
    broot.put_child(2, tree.allocate_node());
    broot.get_child(tree, 2).flags = 1;
    broot.get_child(tree, 2).colour = [200, 100, 100];
    broot.put_child(1, tree.allocate_node());
    broot.get_child(tree, 1).flags = 1;
    broot.get_child(tree, 1).colour = [100, 200, 100];
    broot.put_child(5, tree.allocate_node());
    broot.get_child(tree, 5).flags = 1;
    broot.get_child(tree, 5).colour = [100, 100, 200];
    broot.flags = 0;
    broot.colour = [200, 200, 200];*/

    let mut campos = new_vec(0.0, 0.0, 0.0);
    let mut camrot = new_vec(-10.0, 40.0, 0.0);
    fn get_rotation(camrot: Vector3<f64>) -> Quaternion<f64> {
        use cgmath::Rotation3;
        Quaternion::<f64>::from_axis_angle(new_vec(0.0, 1.0, 0.0), Deg(camrot.y))
            * Quaternion::<f64>::from_axis_angle(new_vec(1.0, 0.0, 0.0), Deg(camrot.x))
    }

    let cubepos = new_vec(0.0, 0.0, 0.0);
    /*
        let mut cpurend = CpuRenderer::new(WIDTH, HEIGHT, RPP/10, RBC, 1);
        cpurend.scenes[0] = VoxelScene {
            tree: tree,
            root: 1,
            camera_position: campos,
            camera_get_rotation(camrot): get_rotation(camrot),
        };

        use vulkano::buffer::CpuBufferPool;

        let cpubufferpool = CpuBufferPool::upload(device.clone());
    */
    let vbuffer = unsafe {
        let mut bufusage = BufferUsage::none();
        bufusage.storage_buffer = true; //TODO potensial BufferUsage::storage_buffer()?
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            bufusage,
            false,
            (0..10000)
                .into_iter()
                .map(|i| *(treeother.base as *const VoxelNode as *const u32).offset(i)),
        )
        .unwrap()
    };

    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        // The alpha mode indicates how the alpha value of the final image will behave. For example
        // you can choose whether the window.window will be opaque or transparent.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // The dimensions of the window.window, only used to initially setup the swapchain.
        // NOTE:
        // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
        // swapchain size must use these dimensions.
        // These dimensions are always the same as the window.window dimensions
        //
        // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
        // These drivers will allow anything but the only sensible value is the window.window dimensions.
        //
        // Because for both of these cases, the swapchain needs to be the window.window dimensions, we just use that.
        let dimensions: [u32; 2] = [
            window.window.get_size().0 as u32,
            window.window.get_size().1 as u32,
        ];

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
    use vulkano::image::immutable::ImmutableImageInitialization;
    use vulkano::image::AttachmentImage;
    use vulkano::image::ImageLayout;
    use vulkano::image::ImageUsage;
    use vulkano::image::ImmutableImage;

    let mut _usage = ImageUsage::none();
    _usage.sampled = true;
    _usage.transfer_destination = true;
    _usage.transfer_source = true;

    let reader =
        std::io::BufReader::new(std::fs::File::open("noise.png").expect("can't find noise.png"));

    let blue_noise_src = image::load(reader, image::ImageFormat::Png).unwrap();

    let (blue_noise, _write) = ImmutableImage::uninitialized(
        device.clone(),
        Dimensions::Dim2d {
            width: blue_noise_src.as_rgba8().unwrap().width() as u32,
            height: blue_noise_src.as_rgba8().unwrap().height() as u32,
        },
        format::R8G8B8A8Unorm,
        1,
        _usage,
        ImageLayout::TransferDstOptimal,
        vec![queue.family()],
    )
    .unwrap();

    let _imgbuffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_source(),
        false,
        blue_noise_src
            .as_rgba8()
            .unwrap()
            .pixels()
            .map(|x| [x[0] as u8, x[1] as u8, x[2] as u8, x[3] as u8] as [u8; 4]),
    )
    .unwrap();

    let cmdbuf = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue_family)
        .unwrap()
        .copy_buffer_to_image(_imgbuffer, _write)
        .unwrap()
        .build()
        .unwrap();

    cmdbuf.execute(queue.clone());

    let rayimages: Vec<Arc<AttachmentImage<format::R8G8B8A8Srgb>>> = (0..images.len())
        .into_iter()
        .map(|x| {
            AttachmentImage::with_usage(
                device.clone(),
                [WIDTH as u32, HEIGHT as u32],
                format::R8G8B8A8Srgb,
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
                    depth: RPP_buffer as u32,
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
    /*
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
    */
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

    let bloomimages: Vec<Arc<StorageImage<Format>>> = (0..images.len())
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

    let bloomimages2: Vec<Arc<StorageImage<Format>>> = (0..images.len())
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
            // Use a resizable viewport set to draw over the entire window.window
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
#define VBUFFER_SIZE 10000

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(set = 0, binding = 0, rgba16) uniform writeonly image3D img;
layout(set = 0, binding = 1) uniform sampler2D noise;
layout(set = 0, binding = 2) buffer VoxelBuffer {
    vec4[] data;
};

layout(push_constant) uniform pushConstants {
    vec3 campos;
    vec3 right;
    vec3 up;
    vec3 forward;
    uint framenum;
} pc;

vec4 rand(vec2 co){
    return texture(noise, co);
}
vec4 rand2(vec2 co, uint x, uint y, uint maxx, uint maxy){
    return rand((vec2(x,y) / vec2(maxx, maxy)) + co);
}

vec3 reflectvec(vec3 v, vec3 normal){
    return v + (-2.0 * dot(v, normal) * normal);
}

const float F0 = 0.04;
float fresnel(vec3 v, vec3 normal){
    return F0 + ((1.0 - F0) * pow(1.0 - dot(normal,v),5.0));
}

vec3 rand_dir_from_surf(vec3 normal, vec2 co){
    vec3 v = rand(co).xyz;
    v *= 2.0;
    v -= vec3(1.0);

    if(dot(v, normal) < 0.0) {
        v += (-2.0 * dot(v, normal) * normal);
    }
    normalize(v);
    return v;
}

vec3 oct_byte_to_vec(uint byte){ // 204 102 240
    return vec3(float(bool(byte & 1)), float(bool(byte & 2)), float(bool(byte & 4))) - vec3(0.5);
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

float slabs(vec3 lb, vec3 rt, vec3 org, vec3 dirfrac) {

    float tx1 = (lb.x - org.x)*dirfrac.x;
    float tx2 = (rt.x - org.x)*dirfrac.x;

    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);

    float ty1 = (lb.y - org.y)*dirfrac.y;
    float ty2 = (rt.y - org.y)*dirfrac.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (lb.z - org.z)*dirfrac.z;
    float tz2 = (rt.z - org.z)*dirfrac.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    float t = tmin * float(tmin > 0) + tmax * float(tmin <= 0);

    return t * float(tmin <= tmax) * float(tmax > 0);
}

struct RayTarget{
    vec3 hitlocation;
    vec3 hitnormal;
    float t;
};

RayTarget cast_ray_v_box(vec3 start, vec3 dir, float size) {
    vec3 p0 = vec3(-0.5) * size;
    vec3 p1 = vec3(0.5) * size;
    vec3 invRaydir = vec3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

    float t = slabs(p0,p1,start,invRaydir);


    vec3 pos = start + dir * t;

    return RayTarget(pos, normalize(biggest_axis(pos)), t);
}

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
    uint em = get_byte(read_vbuffer(ptr + 1),1);
    float f = em / 255.0;
    return (pow(f, 3.4) * 8000.0) * float(em > 0);
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

const uint stacksize = 4;

struct StackFrame {
    uint node;
    float size;
    vec3 pos;
    uint childcount;
    uint children[4];
    uint current_child;
    bool should_init;
};

struct RayResult{
    vec3 hitlocation;
    vec3 hitnormal;
    vec3 raydir;
    float t;
    float specular;
    uint node;
};

RayResult cast_ray_voxel(vec3 ray_start, vec3 ray_dir, uint _node, float _size, vec3 _pos, float specular) {

    vec3 inv_ray_dir = vec3(1.0) / ( ray_dir +
                            vec3((float(ray_dir.x >= 0.0) * EPSILON) - (float(ray_dir.x < 0.0) * EPSILON),
                            (float(ray_dir.y >= 0.0) * EPSILON) - (float(ray_dir.y < 0.0) * EPSILON),
                            (float(ray_dir.z >= 0.0) * EPSILON) - (float(ray_dir.z < 0.0) * EPSILON)));

uint vmask = uint(ray_dir.x < 0.0)
                | (uint(ray_dir.y < 0.0) << 1)
                | (uint(ray_dir.z < 0.0) << 2);

vec3 vmask_1 = vec3((vmask & 1) !=0, (vmask & 2) != 0, (vmask & 4) != 0);
vec3 vmask_0 = vec3((vmask & 1) ==0, (vmask & 2) == 0, (vmask & 4) == 0);

    int stackindex = 0;
    StackFrame stack[stacksize];

    stack[0] = StackFrame(
        _node,
        _size / 2.0,
        _pos,
        0,
        uint[4](0,0,0,0),
        0,
        true
    );



    while(stackindex >= 0 && stackindex < stacksize) {

        if(!stack[stackindex].should_init) {
if(stack[stackindex].current_child >= stack[stackindex].childcount) {
            stackindex -= 1;
            continue;
        }

        uint current_child = stack[stackindex].current_child;

        stack[stackindex + 1].node = get_voxel_child(stack[stackindex].node, stack[stackindex].children[current_child]);

        stack[stackindex + 1].size = stack[stackindex].size / 2.0;

        stack[stackindex + 1].pos = stack[stackindex].pos
            + (oct_byte_to_vec(stack[stackindex].children[current_child])
                * stack[stackindex].size);
        stack[stackindex + 1].should_init = true;

        stack[stackindex].current_child += 1;
        stackindex += 1;

        continue;
        }
            uint root_node = stack[stackindex].node;

            vec3 box_min = vec3(
                -stack[stackindex].size
            ) + stack[stackindex].pos;
            vec3 box_max = vec3(
                stack[stackindex].size
            ) + stack[stackindex].pos;


            vec3 s_vmin = (box_min - ray_start) * inv_ray_dir;

            vec3 s_vmax = (box_max - ray_start) * inv_ray_dir;

            vec3 s_lv = (vmask_1 * s_vmax) +
                        (vmask_0 * s_vmin);

            vec3 s_uv = (vmask_1 * s_vmin) +
                        (vmask_0 * s_vmax);

            float s_lmax = max(s_lv.x, max(s_lv.y, s_lv.z));
            float s_umin = min(s_uv.x, min(s_uv.y, s_uv.z));

            bool is_hit = s_lmax < s_umin;

            float distance = (float(uint(s_lmax > 0.0)) * s_lmax
                + float(uint(s_lmax < 0.0)) * s_umin)
                * float(uint(is_hit));

            vec3 box_mid = (box_min + box_max) / 2.0;

            if(!is_hit){
                stackindex -= 1;
                continue;
            }

            if((get_voxel_flags(root_node) & 1) != 0 && distance > 0.0) {
                vec3 hitlocation = ray_start + (ray_dir * distance);
                return RayResult(
                    hitlocation,
                    normalize(biggest_axis(hitlocation - box_mid)),
                    ray_dir,
                    distance,
                    specular,
                    root_node
                );
            }

            vec3 s_vmid = (box_mid - ray_start) * inv_ray_dir;

            uint[3] masklist = uint[3](1, 2, 4);
            float[3] vallist = float[3](s_vmid.x, s_vmid.y, s_vmid.z);
                if(vallist[0] > vallist[1]) {
                    float temp = vallist[1];
                    vallist[1] = vallist[0];
                    vallist[0] = temp;
                    uint temp2 = masklist[1];
                    masklist[1] = masklist[0];
                    masklist[0] = temp2;
                }
                if(vallist[1] > vallist[2]) {
                    float temp = vallist[2];
                    vallist[2] = vallist[1];
                    vallist[1] = temp;
                    uint temp2 = masklist[2];
                    masklist[2] = masklist[1];
                    masklist[1] = temp2;

                    if(vallist[0] > vallist[1]) {
                        float temp3 = vallist[1];
                        vallist[1] = vallist[0];
                        vallist[0] = temp3;

                        uint temp4 = masklist[1];
                        masklist[1] = masklist[0];
                        masklist[0] = temp4;
                    }
                }

            uint childmask = uint(s_vmid.x < s_lmax)
                | (uint(s_vmid.y < s_lmax) << 1)
                | (uint(s_vmid.z < s_lmax) << 2);

            uint lastmask = uint(s_vmid.x < s_umin)
                | (uint(s_vmid.y < s_umin) << 1)
                | (uint(s_vmid.z < s_umin) << 2);

            uint childcount = 0;
            uint[4] children; //
            uint _child = childmask ^ vmask;
            if((get_voxel_childmask(root_node) & (1 << _child)) != 0) {
                children[0] = _child;
                childcount += 1;
            }

            for(uint x = 0; x < 3; x++){
                if((childmask & masklist[x]) == 0){
                    childmask |= masklist[x];
                    uint child = childmask ^ vmask;
                    if((get_voxel_childmask(root_node) & (1 << child)) == 0){
                        continue;
                    }
                    children[childcount] = child;
                    childcount += 1;
                    if(childmask == lastmask){
                        break;
                    }
                }
            }
            stack[stackindex].childcount = childcount;
            stack[stackindex].children = children;
            stack[stackindex].current_child = 0;
            stack[stackindex].should_init = false;
    }

    return RayResult(
        vec3(0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0),
        0.0,
        0,
        0
    );
}

const uint RBC = 4; // + 1


vec3 calculate_colour(RayResult t, vec3 nc, float ns){
    float specular = ns * ((fresnel(-t.raydir, t.hitnormal) * (1.0 - get_voxel_roughness(t.node))) * ((0.5 * (1.0 - get_voxel_metalness(t.node))) + (0.98 * get_voxel_metalness(t.node))));
    vec3 specular_highlight = ((vec3(0.5) * (1.0 - get_voxel_metalness(t.node)))
            + (get_voxel_colour(t.node) * get_voxel_metalness(t.node)))
            * specular;

    vec3 colour = vec3(0.0);

    colour += specular_highlight * nc;
    colour += (1.0 - ns) * 0.5 * (1.0 - get_voxel_metalness(t.node)) * get_voxel_colour(t.node) * nc;

    //colour *= 1.0 - float(get_voxel_emission(t.node) > 0.0);
    colour += get_voxel_colour(t.node) * get_voxel_emission(t.node);

    colour *= length(t.hitnormal);

//colour = vec3(t.t);

    return colour;
}

void main() {
    vec2 coords = vec2(gl_GlobalInvocationID.xy) / vec2(imageSize(img).xy);
    vec2 pixelsize = vec2(1.0) / vec2(imageSize(img));

    vec3 colour = vec3(0.0);

    RayResult groups[RBC];


        vec2 randcoord = coords + (pixelsize * rand2(coords, gl_GlobalInvocationID.z, pc.framenum, imageSize(img).z, 100).xy);
        vec3 raydir = ((randcoord.x * 2.0 - 1.0) * vec3(pc.right)) + ((randcoord.y * 2.0 - 1.0) * vec3(pc.up)) + vec3(pc.forward);
        vec3 start =  vec3(pc.campos);

        groups[0] = cast_ray_voxel(start, normalize(raydir), 1, 1.0, vec3(0.0), 0);

        if(length(groups[0].hitnormal) < EPSILON){
            imageStore(img, ivec3(gl_GlobalInvocationID.xyz), vec4(0.0,0.0,0.0,1.0));
            return;
        }

        for(uint g = 1; g < RBC; g++){
            float roughness = get_voxel_roughness(groups[g-1].node);
            float specular= round(float(rand2(coords * g, gl_GlobalInvocationID.z, pc.framenum + 2 * g, imageSize(img).z, 102).w > (0.5 * (1.0 - get_voxel_metalness(groups[g-1].node)))));

            vec3 newdir=rand_dir_from_surf(groups[g-1].hitnormal, rand2(rand(randcoord).xz, gl_GlobalInvocationID.z, pc.framenum + 3 * g, imageSize(img).z, 103).yw);

            vec3 specdir=(roughness * newdir) + ((1.0 - roughness) * reflectvec(groups[g-1].raydir, groups[g-1].hitnormal));
            groups[g] = cast_ray_voxel(groups[g-1].hitlocation + ((EPSILON * 2) * groups[g-1].hitnormal), normalize((newdir * (1.0-specular)) + (specdir * specular)), 1, 1.0, vec3(0.0), specular);
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

    imageStore(img, ivec3(gl_GlobalInvocationID.xyz), vec4(colour, min(max(biggest / 8000.0, 0.0), 1.0)));
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

void main() {
    vec3 final = vec3(0.0);

    for(uint i = 0; i < imageSize(img).z; i++){
        vec4 colour = imageLoad(img, ivec3(gl_GlobalInvocationID.xy, i));
        final += colour.xyz * (colour.w * 8000.0) / imageSize(img).z;
    }

    float alpha = max(final.x, max(final.y, final.z));

    final /= alpha;

    imageStore(img2, ivec2(gl_GlobalInvocationID.xy), vec4(final, alpha / 8000.0));
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
    vec4 l1 = imageLoad(imgin, ivec2(gl_GlobalInvocationID.xy));
    vec3 final = l1.xyz * l1.w * 8000.0 * (float(pc.rpp) / float(pc.rpp + pc.so_far_rpp));

    vec4 l2 = imageLoad(imgout, ivec2(gl_GlobalInvocationID.xy));
    final += l2.xyz * l2.w * 8000.0 * (float(pc.so_far_rpp) / float(pc.rpp + pc.so_far_rpp));

    float alpha = max(final.x, max(final.y, final.z));
    final /= alpha;

    imageStore(imgout, ivec2(gl_GlobalInvocationID.xy), vec4(final, alpha / 8000.0));
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
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D imgbloom;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D imgout;

void main() {
    vec4 f = imageLoad(imgin, ivec2(gl_GlobalInvocationID.xy));
    vec3 final = (f.xyz * f.w * 8000.0);

    float biggest = max(1.0, max(final.x, max(final.y, final.z)));
    vec3 true_colour = final / biggest;
    float rel_lum = final.x * 0.2126 + final.y * 0.7152 + final.z * 0.0722;

    float whiteness = min(max(rel_lum - 1.0, 0.0), 1.0);
    vec3 sfinal = mix(final, vec3(1.0), whiteness);
    sfinal.x = min(max(0.0, sfinal.x), 1.0);
    sfinal.y = min(max(0.0, sfinal.y), 1.0);
    sfinal.z = min(max(0.0, sfinal.z), 1.0);

    imageStore(imgout, ivec2(gl_GlobalInvocationID.xy), vec4(sfinal, 1.0));
    imageStore(imgbloom, ivec2(gl_GlobalInvocationID.xy), vec4(true_colour * pow(min(max(rel_lum - 1.0, 0.0), 8000.0) / 8000.0, 0.4), 1.0));
}
"
        }
    }

    mod cs5 {
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
layout(set = 0, binding = 0, rgba8) uniform readonly image2D imgbloom;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D imgbloom2;

layout(push_constant) uniform pushConstants {
    int size;
} pc;

void main() {
    vec3 value = vec3(0.0);
    float count = 0;
    for(int i = -pc.size; i <= pc.size; i++){
        int x = i + int(gl_GlobalInvocationID.x);
        if(x > 0 && x < imageSize(imgbloom).x){
            float weight = (int(pc.size) - abs(i)) / 5;
            value += imageLoad(imgbloom, ivec2(x, gl_GlobalInvocationID.y)).xyz * weight;
            count += weight;
        }
    }
    value /= count;
    imageStore(imgbloom2, ivec2(gl_GlobalInvocationID.xy), vec4(value, 1.0));
}
"
        }
    }

    mod cs6 {
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
layout(set = 0, binding = 0, rgba8) uniform readonly image2D imgbloom;
layout(set = 0, binding = 1, rgba8) uniform image2D imgout;

layout(push_constant) uniform pushConstants {
    int size;
} pc;

void main() {
    vec3 value = vec3(0.0);
    float count = 0;
    for(int i = -pc.size; i <= pc.size; i++){
        int y = i + int(gl_GlobalInvocationID.y);
        if(y > 0 && y < imageSize(imgbloom).y){
            float weight = (int(pc.size) - abs(i)) / 5;
            value += imageLoad(imgbloom, ivec2(gl_GlobalInvocationID.x, y)).xyz * weight;
            count += weight;
        }
    }
    value /= count;
    imageStore(imgout, ivec2(gl_GlobalInvocationID.xy), vec4(value * 4 +
           imageLoad(imgout, ivec2(gl_GlobalInvocationID.xy)).xyz, 1.0));
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

    let shader5 = cs5::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline5 = Arc::new(
        ComputePipeline::new(device.clone(), &shader5.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let shader6 = cs6::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline6 = Arc::new(
        ComputePipeline::new(device.clone(), &shader6.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    use vulkano::descriptor::descriptor_set::DescriptorSet;
    let _soft_repeat = Sampler::new(
        device.clone(),
        Filter::Nearest,
        Filter::Nearest,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();

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
            .add_sampled_image(blue_noise.clone(), _soft_repeat.clone())
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
            .add_image(bloomimages[i].clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }
    let mut set5 = Vec::new();
    for i in 0..images.len() {
        set5.push(Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline5
                    .layout()
                    .descriptor_set_layout(0)
                    .unwrap()
                    .clone(),
            )
            .add_image(bloomimages[i].clone())
            .unwrap()
            .add_image(bloomimages2[i].clone())
            .unwrap()
            .build()
            .unwrap(),
        ));
    }
    let mut set6 = Vec::new();
    for i in 0..images.len() {
        set6.push(Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline6
                    .layout()
                    .descriptor_set_layout(0)
                    .unwrap()
                    .clone(),
            )
            .add_image(bloomimages2[i].clone())
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

    let right = get_rotation(camrot)
        * Vector3::<f64> {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
    let up = get_rotation(camrot)
        * Vector3::<f64> {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
    let forward = get_rotation(camrot)
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

    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    use std::time::Instant;

    let mut snapframenum: usize = 0;
    let mut framenum: u32 = 0;
    let mut framecounter: u32 = 0;
    let mut lastframetimeprintout = Instant::now();

    let dumpimage = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
        },
        vulkano::format::R16G16B16A16Unorm,
        Some(queue.family()),
    )
    .unwrap();

    let buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..WIDTH * HEIGHT * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    while !window.window.should_close() {
        window.poll();

        if snapcount > 0 {
            snapcount -= 1;
            snapframenum += 1;
        } else {
            if window.window.get_key(glfw::Key::O) == glfw::Action::Press {
                snapcount = SNAP_LENGTH;
                snapframenum = 0;
                snaptime = Utc::now();
                std::fs::create_dir_all(snaptime.to_rfc3339());
            }
        }

        let rpp_multiples = {
            if snapcount > 0 {
                SNAP_RPP_mult
            } else {
                RPP_mult
            }
        };

        framenum += 1;
        if framenum > 99 * rpp_multiples {
            framenum = 0;
        }

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Whenever the window.window resizes we need to recreate everything dependent on the window.window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if recreate_swapchain {
            // Get the new dimensions of the window.window.
            let dimensions: [u32; 2] = [
                window.window.get_size().0 as u32,
                window.window.get_size().1 as u32,
            ];

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.window.
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
        // the window.window resizes, but it may not cause the swapchain to become out of date.
        if suboptimal {
            recreate_swapchain = true;
        }

        // Specify the color to clear the framebuffer with i.e. blue ish sorta really actually black
        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

        let mouse_delta = window.get_mouse_delta();
        camrot += new_vec(-mouse_delta.y, mouse_delta.x, 0.0) / 2.0;
        let mut x = camrot.x;
        if x > 90.0 {
            x = 90.0;
        } else if x < -90.0 {
            x = -90.0;
        }
        camrot = new_vec(x, camrot.y, camrot.z);

        let right = get_rotation(camrot)
            * Vector3::<f64> {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            };
        let down = get_rotation(camrot)
            * Vector3::<f64> {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            };
        let forward = get_rotation(camrot)
            * Vector3::<f64> {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            };

        if window.window.get_key(glfw::Key::W) == glfw::Action::Press {
            campos += forward / 1000.0;
        }
        if window.window.get_key(glfw::Key::S) == glfw::Action::Press {
            campos -= forward / 1000.0;
        }
        if window.window.get_key(glfw::Key::D) == glfw::Action::Press {
            campos += right / 1000.0;
        }
        if window.window.get_key(glfw::Key::A) == glfw::Action::Press {
            campos -= right / 1000.0;
        }
        if window.window.get_key(glfw::Key::Space) == glfw::Action::Press {
            campos -= down / 1000.0;
        }
        if window.window.get_key(glfw::Key::LeftShift) == glfw::Action::Press {
            campos += down / 1000.0;
        }

        if window.window.get_key(glfw::Key::B) == glfw::Action::Press {
            campos = new_vec(
                -0.23391431784279354,
                -0.3425462427036379,
                -0.23932050565719531,
            );
            camrot = new_vec(-72.5, 403.0, 0.0);
        }

        /*
        cpurend.scenes[0] = VoxelScene {
            tree: tree,
            root: 1,
            camera_position: campos,
            camera_get_rotation(camrot): get_rotation(camrot),
        };

        let cpubuffer = cpurend.finish_render(
            instance.clone(),
            device.clone(),
            queue.clone(),
            cpubufferpool.clone(),
            cpuimages[image_num].clone(),
            0,
        );
         */

        let mut tempsets = Vec::new(); /*
                                                                          tempsets.push(
                                                                              set3pool
                                                                                  .next()
                                                                                  .add_image(cpuimages[image_num].clone())
                                                                                  .unwrap()
                                                                                  .build()
                                                                                  .unwrap(),
                                       );*/

        let mut rppsets = Vec::new();
        //rppsets.push(cpurend.rpp);

        for i in 0..rpp_multiples {
            rppsets.push(RPP_buffer);
            tempsets.push(
                set3pool
                    .next()
                    .add_image(imagestep2[image_num].clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            );
        }
        let mut sofar_rpp = 0;

        let mut command_buffer_build =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap();

        while rppsets.len() > 0 {
            let rpp = rppsets.pop().unwrap();
            command_buffer_build = command_buffer_build
                .clear_color_image(
                    imagestep[image_num].clone(),
                    vulkano::format::ClearValue::Float([0.0; 4]),
                )
                .unwrap()
                .clear_color_image(
                    imagestep2[image_num].clone(),
                    vulkano::format::ClearValue::Float([0.0; 4]),
                )
                .unwrap()
                .dispatch(
                    [
                        (WIDTH / 8) as u32,
                        (HEIGHT / 8) as u32,
                        (RPP_buffer / 8) as u32,
                    ],
                    compute_pipeline.clone(),
                    (sets[image_num].clone()),
                    gen_push_const(campos, right, down, forward, framenum),
                )
                .unwrap()
                .dispatch(
                    [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                    compute_pipeline2.clone(),
                    (set2[image_num].clone()),
                    (),
                )
                .unwrap()
                .copy_image(
                    imagestep2[image_num].clone(),
                    [0; 3],
                    0,
                    0,
                    dumpimage.clone(),
                    [0; 3],
                    0,
                    0,
                    [WIDTH as u32, HEIGHT as u32, 1],
                    1,
                )
                .unwrap()
                .dispatch(
                    [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                    compute_pipeline3.clone(),
                    (tempsets.pop().unwrap(), set3[image_num].clone()),
                    (rpp as u32, sofar_rpp as u32),
                )
                .unwrap();
            sofar_rpp += rpp;
            framenum += 1;
        }
        command_buffer_build = command_buffer_build
            .dispatch(
                [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                compute_pipeline4.clone(),
                (set4[image_num].clone()),
                (),
            )
            .unwrap()
            .dispatch(
                [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                compute_pipeline5.clone(),
                (set5[image_num].clone()),
                (WIDTH / 12),
            )
            .unwrap()
            .dispatch(
                [(WIDTH / 8) as u32, (HEIGHT / 8) as u32, 1],
                compute_pipeline6.clone(),
                (set6[image_num].clone()),
                (WIDTH / 12),
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
            .unwrap();

        if snapcount > 0 {
            command_buffer_build = command_buffer_build
                .copy_image_to_buffer(rayimages[image_num].clone(), buf.clone())
                .unwrap();
        }

        let command_buffer = command_buffer_build
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
                if snapcount > 0 {
                    future.wait(None).unwrap();

                    let buffer_content = buf.read().unwrap();
                    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
                        WIDTH as u32,
                        HEIGHT as u32,
                        &buffer_content[..],
                    )
                    .unwrap();

                    image
                        .save(format!("{}/{}.png", snaptime.to_rfc3339(), snapframenum))
                        .unwrap();
                }
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
                "Avg frame time: {}, Snap frames left: {}, Camera Position: {}, {}, {}, Camera Rotation: {}, {}, {}",
                (Instant::now() - lastframetimeprintout).as_millis() as f64 / framecounter as f64,
                snapcount,
                campos.x,
                campos.y,
                campos.z,
                camrot.x,
                camrot.y,
                camrot.z,
            );
            framecounter = 0;
            lastframetimeprintout = Instant::now();
        }
    }
}
