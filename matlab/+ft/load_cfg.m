function cfg = load_cfg(root, opt)
%FT.LOAD_CFG  config/rig.json, model.json, gains.json (shared with the Python version) + overrides
%   opt.set = {'path.v_max=30', 'cross.x.Kp=2.2'} overrides gains; opt.model / opt.gains choose other files.
cfg.rig = jsondecode(fileread(fullfile(root, 'config', 'rig.json')));
mf = fullfile(root, 'config', 'model.json'); if isfield(opt, 'model') && ~isempty(opt.model), mf = opt.model; end
gf = fullfile(root, 'config', 'gains.json'); if isfield(opt, 'gains') && ~isempty(opt.gains), gf = opt.gains; end
cfg.model = jsondecode(fileread(mf));
cfg.gains = jsondecode(fileread(gf));
sets = opt.set; if ischar(sets) || isstring(sets), sets = cellstr(sets); end
for i = 1:numel(sets)
    kv = split(string(sets{i}), '=');
    keys = cellstr(split(kv(1), '.'));
    cfg.gains = setfield(cfg.gains, keys{:}, jsondecode(char(kv(2)))); %#ok<SFLD>
end
end
