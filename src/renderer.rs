mod texture;
pub mod buffer;

use std::{borrow::Cow, mem::size_of, sync::Arc, time::{Duration, Instant}};

use bytemuck_derive::{Pod, Zeroable};
use cgmath::{num_traits::Pow, Matrix4, SquareMatrix};
use circular_buffer::CircularBuffer;
use imgui::{Context, FontSource};
use itertools::Itertools;
use wgpu::{
  Backends, BlendState, BufferDescriptor, Color, ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompareFunction, DepthBiasState, DepthStencilState, Device, DeviceDescriptor, Face, FragmentState, FrontFace, IndexFormat, Instance, InstanceDescriptor, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, StencilState, StoreOp, Surface, SurfaceConfiguration, TextureUsages, TextureView, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode
};

use winit::{window::Window, dpi::PhysicalSize};

use crate::{renderer::{buffer::{GenericBufferType, UniformBufferUsage}, texture::Texture}, util::FPVector, world::chunk::ChunkVertex, ArcWorld};

use imgui_winit_support::{WinitPlatform, HiDpiMode};

use self::buffer::{ArrayBuffer, UniformBuffer};

const FPS_ROLLING_AVG: usize = 8; //remember to change both at the same time
const FPS_ROLLING_AVG_F32: f32 = 8.0;

pub struct Renderer {
  surface: Surface,
  surface_cfg: SurfaceConfiguration,
  device: Arc<Device>,
  queue: Arc<Queue>,
  camera_buffer: UniformBuffer<CameraUniform>,
  camera_fragment_buffer: UniformBuffer<CameraFragmentUniform>,
  depth_texture: Texture,
  terrain_pipeline: RenderPipeline,
  sky_pipeline: RenderPipeline,
  sky_vertex_buffer: ArrayBuffer<SkyVertex>,
  sky_camera_buffer: UniformBuffer<SkyCameraUniform>,
  sky_fragment_buffer: UniformBuffer<SkyFragmentUniform>,
  size: PhysicalSize<u32>,
  world: Option<ArcWorld>,
  imgui: RendererImgui,
  last_frame: Option<Instant>,
  frame_times: CircularBuffer<FPS_ROLLING_AVG, Duration>,
}

pub struct RendererImgui {
  ui: Context,
  renderer: imgui_wgpu::Renderer,
  platform: WinitPlatform,
}

struct ImguiData {
  pub fps: Option<f32>,
  pub player_pos: FPVector,
}

//Modified from https://sotrh.github.io/learn-wgpu/
impl Renderer {
  
  pub async fn new(window: &Window) -> Result<Self, RendererCreateError> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        ..Default::default() 
    });
    let surface = unsafe {instance.create_surface(window).unwrap() };

    let adapter = instance.request_adapter(&RequestAdapterOptions {
      power_preference: PowerPreference::HighPerformance,
      ..Default::default()
    }).await.ok_or(RendererCreateError::NoDeviceFound)?;

    let (device, queue) = adapter.request_device(&DeviceDescriptor {
      ..Default::default() //TODO fix.
    }, None).await.map_err(|_| RendererCreateError::RequestDeviceError)?;

    
    let (device, queue) = ( //Put device and queue in arc.
      Arc::new(device),
      Arc::new(queue)
    );

    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    
    let surface_cfg = SurfaceConfiguration {
      format: surface_caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap(),
      height: size.height,
      width: size.width,
      present_mode: surface_caps.present_modes[0],
      usage: TextureUsages::RENDER_ATTACHMENT,
      alpha_mode: surface_caps.alpha_modes[0],
      view_formats: Vec::new(),
    };

    //Imgui setup from: https://github.com/Yatekii/imgui-wgpu-rs/blob/master/examples/hello-world.rs
    let imgui = RendererImgui::new(window, &device, &queue, &surface_cfg);


    println!("Using {} for rendering.", adapter.get_info().name);

    /*
    =================================
    TERRAIN SHADER STUFF
    =================================
    */

    let camera_buffer = UniformBuffer::<CameraUniform>::new(&device, UniformBufferUsage::Vertex, Some("Vertex camera buffer"));
    let camera_fragment_buffer = UniformBuffer::<CameraFragmentUniform>::new(&device, UniformBufferUsage::Fragment, Some("Fragment camera buffer"));

    
    //Load world terrain shader module.
    let terrain_shader = include_str!("../shaders/terrain.wgsl");
    let terrain_shader_module = device.create_shader_module(ShaderModuleDescriptor { 
      label: Some("World terrain shader module."), 
      source: ShaderSource::Wgsl(Cow::from(terrain_shader))
    });    
    let depth_texture = Texture::create_depth_texture(&device, &surface_cfg, "Depth texture and stuff");

    let terrain_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      bind_group_layouts: &[&camera_buffer.get_bind_group_layout(), &camera_fragment_buffer.get_bind_group_layout()],
      label: Some("pipeline layout"),
      push_constant_ranges: &[]
    });

    let terrain_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
      depth_stencil: Some(DepthStencilState {
        format: Texture::DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    }),
      vertex: VertexState {
        buffers: &[ChunkVertex::desc()],
        entry_point: "vs_main",
        module: &terrain_shader_module
      },
      fragment: Some(FragmentState {
        entry_point: "fs_main",
        module: &terrain_shader_module,
        targets: &[Some(ColorTargetState {
          blend: Some(BlendState::REPLACE),
          format: surface_cfg.format,
          write_mask: ColorWrites::ALL
        })]
      }),
      label: Some("Very cool pipeline layout."),
      layout: Some(&terrain_pipeline_layout),
      multisample: MultisampleState {
        count: 1, //MSAA?
        mask: !0,
        alpha_to_coverage_enabled: false, //AA
    },
      multiview: None,
      primitive: PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: Some(Face::Back),
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
      },
    });

    /*
    =================================
    SKY SHADER STUFF
    =================================
    */

    let sky_camera_buffer = UniformBuffer::<SkyCameraUniform>::new(&device, UniformBufferUsage::Vertex, Some("Sky vertex camera buffer"));
    let sky_fragment_buffer = UniformBuffer::<SkyFragmentUniform>::new(&device, UniformBufferUsage::Fragment, Some("Sky vertex camera buffer"));

    //Load sky shader module.
    let sky_shader = include_str!("../shaders/sky.wgsl");
    let sky_shader_module = device.create_shader_module(ShaderModuleDescriptor { 
      label: Some("Sun and sky shader module"), 
      source: ShaderSource::Wgsl(Cow::from(sky_shader))
    });

    let sky_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      bind_group_layouts: &[&sky_camera_buffer.get_bind_group_layout(), &sky_fragment_buffer.get_bind_group_layout()],
      label: Some("pipeline layout"),
      push_constant_ranges: &[]
    });

    let sky_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
      depth_stencil: None,
      vertex: VertexState {
        buffers: &[SkyVertex::desc()],
        entry_point: "vs_main",
        module: &sky_shader_module
      },
      fragment: Some(FragmentState {
        entry_point: "fs_main",
        module: &sky_shader_module,
        targets: &[Some(ColorTargetState {
          blend: Some(BlendState::REPLACE),
          format: surface_cfg.format,
          write_mask: ColorWrites::ALL
        })]
      }),
      label: Some("Very cool pipeline layout."),
      layout: Some(&sky_pipeline_layout),
      multisample: MultisampleState {
        count: 1, //MSAA?
        mask: !0,
        alpha_to_coverage_enabled: false, //AA
    },
      multiview: None,
      primitive: PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: None,
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
      },
    });

    let sky_vertex_buffer = ArrayBuffer::new(&device, &queue, GenericBufferType::Vertex, &SKY_VERTICES, 0);

    Ok(Self {
      surface,
      surface_cfg,
      device,
      queue,
      depth_texture,
      camera_buffer,
      camera_fragment_buffer,
      terrain_pipeline,
      sky_pipeline,
      sky_vertex_buffer,
      sky_camera_buffer,
      sky_fragment_buffer,
      size,
      world: None,
      imgui,
      last_frame: None,
      frame_times: CircularBuffer::new()
    })
  }

  pub fn get_device_queue(&self) -> (Arc<Device>, Arc<Queue>) {
    (self.device.clone(), self.queue.clone())
  }

  pub fn resize(&mut self, size: PhysicalSize<u32>) {
    if size.width > 32 && size.height > 32 {
      self.size = size;
      self.surface_cfg.width = size.width;
      self.surface_cfg.height = size.height;
      self.surface.configure(&self.device, &self.surface_cfg);
      self.depth_texture = Texture::create_depth_texture(&self.device, &self.surface_cfg, "another depth texture");
    }
  }

  pub fn bind_world(&mut self, world: ArcWorld) {
    self.world = Some(world);
  }

  pub fn imgui(&mut self) -> &mut RendererImgui {
    &mut self.imgui
  }

  pub fn render(&mut self) -> Result<(), RenderError> {
    

    let world = match self.world.as_ref() {
        Some(w) => w,
        None => {
          return Err(RenderError::NoWorldError);
        },
    };

    let (view_mat, player_pos, chunk_list, light_data, pos_fpv) = {
      let world_lock = world.lock().unwrap();
      (
        world_lock.get_player_view(self.size.width as f32/self.size.height as f32), 
        world_lock.get_player_pos_c(),
        world_lock.get_terrain().get_meshes(),
        world_lock.get_daylight_data(),
        world_lock.get_player_pos(),
      )
    };

    let camera_view = CameraUniform {
      view: view_mat.into(),
      position_abs: player_pos.block_int.into(),
      position_rel: player_pos.block_dec.into(),
      padding_1: 0u32,
      sun_intensity: light_data.light_level,
      padding_2: 0u32,
      sun_normal: light_data.sun_direction.into(),
    };
    self.camera_buffer.update(&self.queue, camera_view);

    let camera_fragment_view = CameraFragmentUniform {
      sun_intensity: light_data.light_level,
      sun_normal: light_data.sun_direction.into()
    };
    self.camera_fragment_buffer.update(&self.queue, camera_fragment_view);


    let out = self.surface.get_current_texture().map_err(|_| RenderError::SurfaceError)?;
    let view = out.texture.create_view(&Default::default());
    let depth_view = &self.depth_texture.view;

    
    let sky_camera_uniform = SkyCameraUniform {
      view_matrix_inv: view_mat.invert().unwrap().into()
    };
    self.sky_camera_buffer.update(&self.queue, sky_camera_uniform);

    let sky_fragment_uniform = SkyFragmentUniform {
      sun_direction: light_data.sun_direction.into(),
      light_level: light_data.light_level
    };
    self.sky_fragment_buffer.update(&self.queue, sky_fragment_uniform);

    let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("very cool command encoder")
    });

    
    let sky_buf = self.sky_vertex_buffer.get_buffer();
    { //Clear screen.
      let mut sky_render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Sky render pass."),
        color_attachments: &[Some(RenderPassColorAttachment {
          ops: Operations {
            load: LoadOp::Clear(Color {
              r: 0.0,
              g: 0.0,
              b: 0.0,
              a: 0.0
            }),
            store: StoreOp::Store
          },
          resolve_target: None,
          view: &view
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
      });

      sky_render_pass.set_bind_group(0, &self.sky_camera_buffer.get_bind_group(), &[]);
      sky_render_pass.set_bind_group(1, &self.sky_fragment_buffer.get_bind_group(), &[]);

      sky_render_pass.set_pipeline(&self.sky_pipeline);
      sky_render_pass.set_vertex_buffer(0, sky_buf.slice(..));
      sky_render_pass.draw(0..SKY_VERTICES.len() as u32, 0..1);
    }

    {
      //Filter out empty chunks.
      let chunk_datas = chunk_list.into_iter().filter_map(|(_, data)| if data.index_buffer.1 > 0 {Some(data)} else {None}).collect_vec();

      // let ll = light_data.light_level.pow(2.2) as f64;
      let mut terrain_render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
          ops: Operations {
            // load: LoadOp::Clear(Color {
            //   r: 0.3 * ll,
            //   g: 0.3 * ll,
            //   b: 0.7 * ll,
            //   a: 0.0
            // }),
            load: LoadOp::Load,
            store: StoreOp::Store
          },
          resolve_target: None,
          view: &view
        })],
        label: Some("Screen pass"),
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
          view: &depth_view,
          depth_ops: Some(Operations {
              load: LoadOp::Clear(1.0),
              store: StoreOp::Store,
          }),
          stencil_ops: None,
        }),
        occlusion_query_set: None,
        timestamp_writes: None
      });
      terrain_render_pass.set_bind_group(0, &self.camera_buffer.get_bind_group(), &[]); //Set player and camera uniform./Chunk uniform group
      terrain_render_pass.set_bind_group(1, &self.camera_fragment_buffer.get_bind_group(), &[]); //Set camera data to fragment shader too.

      terrain_render_pass.set_pipeline(&self.terrain_pipeline);

      for data in chunk_datas.iter(){
        terrain_render_pass.set_vertex_buffer(0, data.vertex_buffer.0.slice(..data.vertex_buffer.1 * size_of::<ChunkVertex>() as u64));
        terrain_render_pass.set_index_buffer(data.index_buffer.0.slice(..data.index_buffer.1 * size_of::<u32>() as u64), IndexFormat::Uint32);
        terrain_render_pass.draw_indexed(0..data.index_buffer.1 as u32, 0, 0..1);
      }
      
    }
    
    

    //Calculate fps
    let fps_avg = if self.frame_times.is_full() {
      //Calculate mean only if the fps buffer is full.

      self.frame_times.iter().map(|dur| *dur).reduce(|dur, acc| {
        dur+acc
      }).map(|total_dur| FPS_ROLLING_AVG_F32/total_dur.as_secs_f32())
    } else {None};

    //Process imgui data.
    let data = ImguiData {
        fps: fps_avg,
        player_pos: pos_fpv
    };

    self.imgui.render(&data, &mut encoder, &self.device, &self.queue, &view, &depth_view)?;

    let command_buffers = std::iter::once(encoder.finish());
    self.queue.submit(command_buffers);
    // let imgui_command_buffer = self.imgui.renderer.render(draw_data, queue, device, rpass);
    out.present();

    //FPS tracking.
    if let Some(last_frame_inst) = self.last_frame {
      //Calculate frame time.
      let frame_duration = Instant::now().duration_since(last_frame_inst);
      self.frame_times.push_front(frame_duration);
    }
    self.last_frame = Some(Instant::now());

    //Rendering complete.
    Ok(())
  }
}

impl RendererImgui {
  fn new(window: &Window, device: &Device, queue: &Queue, surface_cfg: &SurfaceConfiguration) -> Self {
    //Setup imgui.
    let mut imgui_ctx = imgui::Context::create();
    imgui_ctx.set_ini_filename(None);

    //Bind imgui I/O to winit.
    let mut imgui_platform = WinitPlatform::init(&mut imgui_ctx);
    imgui_platform.attach_window(imgui_ctx.io_mut(), &window, HiDpiMode::Default);

    let imgui_renderer_cfg = imgui_wgpu::RendererConfig {
        texture_format: surface_cfg.format,
        depth_format: Some(Texture::DEPTH_FORMAT),
        ..Default::default()
    };

    let hidpi_factor = window.scale_factor();

    let font_size = (13.0 * hidpi_factor) as f32;
    imgui_ctx.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    imgui_ctx.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(imgui::FontConfig {
            oversample_h: 1,
            pixel_snap_h: true,
            size_pixels: font_size,
            ..Default::default()
        }),
    }]);

    let imgui_renderer = imgui_wgpu::Renderer::new(&mut imgui_ctx, &device, &queue, imgui_renderer_cfg);
    
    RendererImgui {
      ui: imgui_ctx,
      platform: imgui_platform,
      renderer: imgui_renderer
    }
  }

  pub fn window_event<T>(&mut self, window: &Window, event: &winit::event::Event<'_, T>) {
    self.platform.handle_event(self.ui.io_mut(), window, event);
  }

  fn prep_window(frame: &mut imgui::Ui, data: &ImguiData) {
    let fps_string = data.fps.map_or(String::from("???"), |fps| format!("{:.1}", fps));

    frame.window("Debug Menu")
      .size([300.0, 150.0], imgui::Condition::FirstUseEver)
      .build(|| {
        frame.text_colored([1.0, 1.0, 0.7, 1.0], "Hold ALT to access cursor...");
        frame.text_wrapped(format!("FPS: {}", fps_string));
        frame.text_wrapped(format!("X: {:.4}", {data.player_pos.inner.x}));
        frame.text_wrapped(format!("Y: {:.4}", {data.player_pos.inner.y}));
        frame.text_wrapped(format!("Z: {:.4}", {data.player_pos.inner.z}));
      });
  }


  fn render(&mut self, data: &ImguiData, encoder: &mut CommandEncoder, device: &Device, queue: &Queue, view: &TextureView, depth_view: &TextureView) -> Result<(), RenderError> {
    let frame = self.ui.frame();
    
    Self::prep_window(frame, data);
    // let mut demo_open = true;
    //TODO make debug menu
    // if demo_open {
    //   frame.show_demo_window(&mut demo_open); //testing demo window.
    // }
    
    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Imgui render pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &view,
          resolve_target: None,
          ops: wgpu::Operations {
              load: wgpu::LoadOp::Load,
              store: wgpu::StoreOp::Store,
          },
        })],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
          view: &depth_view,
          depth_ops: Some(Operations {
              load: LoadOp::Load,
              store: StoreOp::Store,
          }),
          stencil_ops: None,
        }), //TODO attach depth buffer to imgui
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    self.renderer.render(self.ui.render(), queue, device, &mut rpass).map_err(|_| RenderError::ImguiError)?;

    Ok(())
    
  }
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniform {
  pub view: [[f32; 4]; 4],
  pub position_abs: [i32; 3], //The player position per block.
  pub padding_1: u32,
  pub position_rel: [f32; 3], //Decimal part of player position. ALREADY ACCOUNTED FOR IN VIEW MATRIX.
  pub padding_2: u32,
  pub sun_normal: [f32; 3],
  pub sun_intensity: f32,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraFragmentUniform {
  pub sun_normal: [f32; 3],
  pub sun_intensity: f32
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct SkyCameraUniform {
  pub view_matrix_inv: [[f32; 4]; 4]
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct SkyFragmentUniform {
  pub sun_direction: [f32; 3],
  pub light_level: f32
}

trait Descriptable {
  fn desc<'a>() -> VertexBufferLayout<'a>;
}

impl Descriptable for ChunkVertex {
    fn desc<'a>() -> VertexBufferLayout<'a> {    
      VertexBufferLayout {
        array_stride: std::mem::size_of::<ChunkVertex>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &[
          VertexAttribute {
            format: VertexFormat::Sint32x3,
            offset: 0,
            shader_location: 0, //Absolute position.
          },
          VertexAttribute { //Relative Position
            format: VertexFormat::Float32x3,
            offset: size_of::<[i32; 3]>() as u64,
            shader_location: 1
          },
          VertexAttribute { //Colour
            format: VertexFormat::Float32x4,
            offset: size_of::<[i32; 3]>() as u64 + size_of::<[f32; 3]>() as u64,
            shader_location: 2
          },
          VertexAttribute { //Vertex normal
            format: VertexFormat::Float32x3,
            offset: size_of::<[i32; 3]>() as u64 + size_of::<[f32; 3]>() as u64 + size_of::<[f32; 4]>() as u64,
            shader_location: 3
          }
        ],
      }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SkyVertex { //Used to draw the sky.
  pub screen_position: [f32; 2]
}

impl Descriptable for SkyVertex {
  fn desc<'a>() -> VertexBufferLayout<'a> {
    VertexBufferLayout {
      array_stride: std::mem::size_of::<SkyVertex>() as u64,
      step_mode: VertexStepMode::Vertex,
      attributes: &[
        VertexAttribute {
          format: VertexFormat::Float32x2,
          offset: 0,
          shader_location: 0
        },
      ],
    }
  }
}

const SKY_VERTICES: [SkyVertex; 6] = [
  SkyVertex {screen_position: [-1.0, 1.0]}, //Top left
  SkyVertex {screen_position: [-1.0, -1.0]}, //Bottom left
  SkyVertex {screen_position: [1.0, 1.0]}, //Top right
  SkyVertex {screen_position: [1.0, 1.0]}, //Top right (again)
  SkyVertex {screen_position: [-1.0, -1.0]}, //Bottom left (again)
  SkyVertex {screen_position: [1.0, -1.0]}, //Bottom right
];

#[derive(Debug)]
pub enum RendererCreateError {
  NoDeviceFound,
  RequestDeviceError,
  // ShaderLoadError,
}

#[derive(Debug)]
pub enum RenderError {
  SurfaceError,
  NoWorldError,
  ImguiError
}
