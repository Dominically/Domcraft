mod texture;
pub mod buffer;

use std::{fs::File, io::Read, borrow::Cow, sync::Arc, mem::size_of};

use bytemuck_derive::{Pod, Zeroable};
use imgui::{Context, FontSource};
use itertools::Itertools;
use wgpu::{
  Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompareFunction, DepthBiasState, DepthStencilState, Device, DeviceDescriptor, Face, FragmentState, FrontFace, IndexFormat, Instance, InstanceDescriptor, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, StoreOp, Surface, SurfaceConfiguration, TextureUsages, TextureView, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode
};

use winit::{window::Window, dpi::PhysicalSize};

use crate::{world::chunk::ChunkVertex, ArcWorld, renderer::texture::Texture};

use imgui_winit_support::{WinitPlatform, HiDpiMode};

pub struct Renderer {
  surface: Surface,
  surface_cfg: SurfaceConfiguration,
  device: Arc<Device>,
  queue: Arc<Queue>,
  camera_buffer: Buffer,
  camera_bind_group: BindGroup,
  depth_texture: Texture,
  pipeline: RenderPipeline,
  size: PhysicalSize<u32>,
  world: Option<ArcWorld>,
  imgui: RendererImgui,
  // imgui_renderer: imgui_wgpu::Renderer,
  // imgui_platform: WinitPlatform,
}

pub struct RendererImgui {
  ui: Context,
  renderer: imgui_wgpu::Renderer,
  platform: WinitPlatform,
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

    let mut shader_file = File::open("./shaders/shader.wgsl").map_err(|_| RendererCreateError::ShaderLoadError)?;
    let mut shader: String = String::new();
    shader_file.read_to_string(&mut shader).map_err(|_| RendererCreateError::ShaderLoadError)?;

    let shader_module = device.create_shader_module(ShaderModuleDescriptor { 
      label: Some("very cool shader module"), 
      source: ShaderSource::Wgsl(Cow::from(shader))
    });

    let camera_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("camera buffer and stuff"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let camera_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
      entries: &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
      label: Some("camera bind group layout"), 
    });

    let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("camera bind group stuff"),
        layout: &camera_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    

    let depth_texture = Texture::create_depth_texture(&device, &surface_cfg, "Depth texture and stuff");

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      bind_group_layouts: &[&camera_bind_group_layout],
      label: Some("pipeline layout"),
      push_constant_ranges: &[]
    }); //Not really necessary right now.

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
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
        module: &shader_module
      },
      fragment: Some(FragmentState {
        entry_point: "fs_main",
        module: &shader_module,
        targets: &[Some(ColorTargetState {
          blend: Some(BlendState::REPLACE),
          format: surface_cfg.format,
          write_mask: ColorWrites::ALL
        })]
      }),
      label: Some("Very cool pipeline layout."),
      layout: Some(&pipeline_layout),
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

    Ok(Self {
      surface,
      surface_cfg,
      device,
      queue,
      depth_texture,
      camera_bind_group,
      camera_buffer,
      pipeline,
      size,
      world: None,
      imgui,
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

    let (view_mat, player_pos, chunk_list, light_data) = {
      let world_lock = world.lock().unwrap();
      (
        world_lock.get_player_view(self.size.width as f32/self.size.height as f32), 
        world_lock.get_player_pos_c(),
        world_lock.get_terrain().get_meshes(),
        world_lock.get_daylight_data()
      )
    };
    let camera_view = CameraUniform {
      view: view_mat.into(),
      player_position: player_pos.block_int.into(),
      padding_1: 0u32,
      sun_intensity: light_data.light_level,
      sun_normal: light_data.sun_direction.into(),
    };
    self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_view]));

    let out = self.surface.get_current_texture().map_err(|_| RenderError::SurfaceError)?;
    let view = out.texture.create_view(&Default::default());
    let depth_view = &self.depth_texture.view;

    let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("very cool command encoder")
    });

    { //Clear screen.
      //Filter out empty chunks.
      let chunk_datas = chunk_list.into_iter().filter_map(|(_, data)| if data.index_buffer.1 > 0 {Some(data)} else {None}).collect_vec();

      let ll = light_data.light_level as f64;
      let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
          ops: Operations {
            load: LoadOp::Clear(Color {
              r: 0.3 * ll,
              g: 0.3 * ll,
              b: 0.7 * ll,
              a: 0.0
            }),
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
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]); //Set player and camera uniform./Chunk uniform group
      render_pass.set_pipeline(&self.pipeline);

      for data in chunk_datas.iter(){
        render_pass.set_vertex_buffer(0, data.vertex_buffer.0.slice(..data.vertex_buffer.1 * size_of::<ChunkVertex>() as u64));
        render_pass.set_index_buffer(data.index_buffer.0.slice(..data.index_buffer.1 * size_of::<u32>() as u64), IndexFormat::Uint32);
        render_pass.draw_indexed(0..data.index_buffer.1 as u32, 0, 0..1);
      }
      
    }

    self.imgui.render(&mut encoder, &self.device, &self.queue, &view, &depth_view)?;

    let command_buffers = std::iter::once(encoder.finish());
    self.queue.submit(command_buffers);
    // let imgui_command_buffer = self.imgui.renderer.render(draw_data, queue, device, rpass);
    out.present();

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
        // sample_count: todo!(),
        // shader: todo!(),
        // vertex_shader_entry_point: todo!(),
        // fragment_shader_entry_point: todo!(),
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

  fn render(&mut self, encoder: &mut CommandEncoder, device: &Device, queue: &Queue, view: &TextureView, depth_view: &TextureView) -> Result<(), RenderError> {
    let frame = self.ui.frame();
    let mut demo_open = true;
    //TODO make debug menu
    //frame.show_demo_window(&mut demo_open); //testing demo window.
    
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

trait Descriptable {
  fn desc<'a>() -> VertexBufferLayout<'a>;
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniform {
  pub view: [[f32; 4]; 4],
  pub player_position: [i32; 3], //The player position per block.
  pub padding_1: u32,
  pub sun_normal: [f32; 3],
  pub sun_intensity: f32,
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

#[derive(Debug)]
pub enum RendererCreateError {
  NoDeviceFound,
  RequestDeviceError,
  ShaderLoadError,
}

#[derive(Debug)]
pub enum RenderError {
  SurfaceError,
  NoWorldError,
  ImguiError
}
