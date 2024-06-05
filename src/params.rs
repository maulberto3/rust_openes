#[derive(Debug)]
pub enum Params {
    OpenES(OpenESParams),
    OtherParams,
}

#[derive(Debug)]
pub struct OpenESParams {
    pub sigma_init: f32,
    pub sigma_decay: f32,
    pub sigma_limit: f32,
    pub init_min: f32,
    pub init_max: f32,
    pub clip_min: f32,
    pub clip_max: f32,
    // optim params
    pub beta_1: f32,
    pub beta_2: f32,
    pub eps: f32,
    pub learning_rate: f32,
}

impl OpenESParams {
    pub fn default_params() -> Self {
        OpenESParams {
            // opt_params
            sigma_init: 1.0,
            sigma_decay: 0.999,
            sigma_limit: 0.04,
            init_min: -1.0,
            init_max: 1.0,
            clip_min: f32::NEG_INFINITY,
            clip_max: f32::INFINITY,
            // optim params
            beta_1: 0.99,
            beta_2: 0.999,
            eps: 1e-8,
            learning_rate: 0.05,
        }
    }
}
