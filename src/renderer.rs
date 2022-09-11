mod texture;
pub mod buffer;

use std::{fs::File, io::Read, borrow::Cow, sync::Arc, mem::size_of};

use bytemuck_derive::{Pod, Zeroable};
use itertools::Itertools;
//these imports are not a joke wtf
use wgpu::{
  Instance,
  Backends,
  RequestAdapterOptions,
  PowerPreference,
  DeviceDescriptor,
  Device,
  Queue,
  BufferUsages,
  VertexBufferLayout,
  VertexAttribute,
  Buffer,
  ShaderModuleDescriptor,
  ShaderSource,
  RenderPipelineDescriptor,
  FragmentState,
  VertexState,
  MultisampleState,
  PipelineLayoutDescriptor,
  PrimitiveState,
  PrimitiveTopology,
  FrontFace,
  Face,
  PolygonMode,
  RenderPipeline,
  Surface,
  ColorTargetState,
  ColorWrites,
  BlendState,
  SurfaceConfiguration,
  PresentMode,
  TextureUsages,
  RenderPassDescriptor,
  RenderPassColorAttachment,
  Operations,
  Color,
  CommandEncoderDescriptor,
  VertexStepMode,
  VertexFormat,
  BufferDescriptor,
  BindGroupLayoutDescriptor,
  BindGroupLayoutEntry,
  ShaderStages,
  BindingType,
  BufferBindingType,
  BindGroupDescriptor,
  BindGroupEntry,
  BindGroup,
  DepthStencilState,
  CompareFunction,
  StencilState,
  DepthBiasState,
  RenderPassDepthStencilAttachment,
  LoadOp,
  IndexFormat
};

use winit::{window::Window, dpi::PhysicalSize};

use crate::{world::chunk::ChunkVertex, ArcWorld, renderer::{texture::Texture, }};

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
  world: Option<ArcWorld>
}


impl Renderer {
  
  pub async fn new(window: &Window) -> Result<Self, RendererCreateError> {
    let instance = Instance::new(Backends::PRIMARY);
    let surface = unsafe {instance.create_surface(window) };

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
    let surface_cfg = SurfaceConfiguration {
      format: surface.get_supported_formats(&adapter)[0],
      height: size.height,
      width: size.width,
      present_mode: PresentMode::Fifo,
      usage: TextureUsages::RENDER_ATTACHMENT
    };
    
    println!("Using {}", adapter.get_info().name);
    

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
      label: Some("stinky pipeline layout"),
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

  pub fn render(&mut self) -> Result<(), RenderError> {
    let world = match self.world.as_ref() {
        Some(w) => w,
        None => {
          return Err(RenderError::NoWorldError);
        },
    };

    let (view_mat, player_pos) = {
      let world_lock = world.lock().unwrap();
      (world_lock.get_player_view(self.size.width as f32/self.size.height as f32), world_lock.get_player_pos())
    };

    let camera_view = CameraUniform {
      view: view_mat.into(),
      player_position: player_pos.block_int.into(),
      sun_intensity: 1.0,
      sun_normal: [-0.20265, 0.97566, 0.08378],
      padding: 0u32
    };
    self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_view]));

    let out = self.surface.get_current_texture().map_err(|_| RenderError::SurfaceError)?;
    let view = out.texture.create_view(&Default::default());

    let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("very cool command encoder")
    });

    let chunk_list = self.world.as_ref().unwrap().lock().unwrap().get_terrain().get_meshes(); //Get list of chunk meshes.

    { //Clear screen.
      //Filter out empty chunks.
      let chunk_datas = chunk_list.into_iter().filter_map(|(_, data)| if data.index_buffer.1 > 0 {Some(data)} else {None}).collect_vec();

      let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
          ops: Operations {
            load: LoadOp::Clear(Color {
              r: 0.3,
              g: 0.3,
              b: 0.7,
              a: 0.0
            }),
            store: true
          },
          resolve_target: None,
          view: &view
        })],
        label: Some("Screen pass"),
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
          view: &self.depth_texture.view,
          depth_ops: Some(Operations {
              load: LoadOp::Clear(1.0),
              store: true,
          }),
          stencil_ops: None,
        }),
      });
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]); //Set player and camera uniform./Chunk uniform group
      render_pass.set_pipeline(&self.pipeline);

      for data in chunk_datas.iter(){
        render_pass.set_vertex_buffer(0, data.vertex_buffer.0.slice(..data.vertex_buffer.1 * size_of::<ChunkVertex>() as u64));
        render_pass.set_index_buffer(data.index_buffer.0.slice(..data.index_buffer.1 * size_of::<u32>() as u64), IndexFormat::Uint32);
        render_pass.draw_indexed(0..data.index_buffer.1 as u32, 0, 0..1);
      }
      
    }

    let command_buffers = std::iter::once(encoder.finish());
    self.queue.submit(command_buffers);
    out.present();

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
  pub sun_normal: [f32; 3],
  pub sun_intensity: f32,
  pub padding: u32 //Make the uniform buffer a size multiple of 8.
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
            format: VertexFormat::Float32x3,
            offset: size_of::<[i32; 3]>() as u64 + size_of::<[f32; 3]>() as u64,
            shader_location: 2
          },
          VertexAttribute { //Vertex normal
            format: VertexFormat::Float32x3,
            offset: size_of::<[i32; 3]>() as u64 + size_of::<[f32; 3]>() as u64 * 2,
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
  NoWorldError
}
