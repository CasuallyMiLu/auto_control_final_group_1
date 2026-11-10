function A = audit_real_sysid(projectRoot, outputDir)
%AUDIT_REAL_SYSID Audit the four 2026-11-10 commissioning flights offline.
% Loads existing models; does not refit or change flight settings. Complete
% well-tracked excitation blocks are compared across runs. Initial conditions
% are estimated per block; source-model training blocks are explicitly tagged.
if nargin < 2
    outputDir = fullfile(projectRoot,'runs','sysid_review_20261110');
end
if ~isfolder(outputDir), mkdir(outputDir); end
stamps = ["171015","172242","174033","174633"];
modelStamps = stamps(3:4);
A = struct('status','preliminary_estimation_complete_validation_not_accepted', ...
    'analysisTs',1/30,'modelRefitted',false, ...
    'note','Cross-run checks use only complete, screened blocks. Below-30-percent battery remains a flagged confounder. Per-block initial states are estimated.');
A.screeningPolicy = 'No range truncation; valid fraction >=0.98; maximum valid-position gap <=0.2 s; bounding-box height ratio <=2.2. These are audit heuristics, not proof of tracking accuracy or flight safety limits.';
models = struct(); prepared = struct();
for stamp = modelStamps
    p = fullfile(projectRoot,'runs',"20261110_"+stamp+"_sysid_circle_matlab_real01");
    D = load(fullfile(p,'matlab_sysid.mat'),'M');
    models.("r"+stamp) = D.M;
end
for n = 1:numel(stamps)
    stamp = stamps(n);
    p = fullfile(projectRoot,'runs',"20261110_"+stamp+"_sysid_circle_matlab_real01");
    meta = jsondecode(fileread(fullfile(p,'meta.json')));
    assert(~meta.sim,'Expected real flight data.');
    op = detectImportOptions(fullfile(p,'log.csv'),'TextType','string');
    op = setvartype(op,{'phase','excite_axis'},'string');
    T = readtable(fullfile(p,'log.csv'),op);
    [blocks,stats] = nominalBlocks(T,meta);
    R = struct('stamp',stamp,'folder',fullfile('runs',"20261110_"+stamp+"_sysid_circle_matlab_real01"),'endReason',meta.land_reason, ...
        'totalRows',height(T),'blocks',stats, ...
        'hasIdentification',isfile(fullfile(p,'matlab_sysid.mat')));
    R.batteryRange = finiteRange(T.bat);
    R.models = struct();
    if R.hasIdentification
        for ax = ["x","y"]
            m = models.("r"+stamp).(ax);
            R.models.(ax) = struct('Kv',m.Kv,'tau',m.tau,'L',m.L, ...
                'nTrain',m.nTrain,'nValid',m.nValid, ...
                'originalTrainP',m.fitTrainP,'originalValidationP',m.fitValidP, ...
                'originalTrainT',m.fitTrainT,'originalValidationT',m.fitValidT);
        end
    end
    A.runs(n) = R;
    prepared.("r"+stamp) = struct('blocks',{blocks},'stats',stats);
end
co = compareOptions('InitialCondition','estimate');
records = struct([]); row = 0;
fig = figure('Color','w','Position',[80 80 1250 850]);
tl = tiledlayout(fig,2,2,'TileSpacing','compact');
for k = 1:2
    names = ["x","y"]; ax = names(k);
    ms = {models.r174033.(ax).sysP,models.r174633.(ax).sysP, ...
        models.r174033.(ax).sysT,models.r174633.(ax).sysT};
    labels = ["174033 P1DI","174633 P1DI","174033 tfest","174633 tfest"];
    for j = 1:4, ms{j}.Name = char(labels(j)); end
    for sourceIndex = 1:numel(stamps)
        stamp = stamps(sourceIndex); D = prepared.("r"+stamp);
        candidates = find(string({D.stats.axis}) == ax & [D.stats.screenedComplete]);
        first = true;
        for b = candidates
            z = D.blocks{b};
            [~,f] = compare(z,ms{:},co);
            if iscell(f), f = cellfun(@(v)v(1),f); end
            row = row+1;
            records(row).sourceStamp = stamp;
            records(row).axis = ax;
            records(row).nominalBlock = D.stats(b).nominalBlock;
            records(row).amplitude = D.stats(b).amplitude;
            records(row).batteryMinimum = D.stats(b).batteryRange(1);
            records(row).belowStartBatteryThreshold = D.stats(b).belowStartBatteryThreshold;
            records(row).sourceModelRole = "external validation";
            if sourceIndex >= 3
                records(row).sourceModelRole = "validation";
            end
            if sourceIndex >= 3 && (D.stats(b).nominalBlock <= 2 || D.stats(b).nominalBlock == 5)
                records(row).sourceModelRole = "training";
            end
            records(row).modelOrder = labels;
            records(row).fits = reshape(f,1,[]);
            rd = struct([]);
            for modelIndex = 1:2
                [~,r] = resid(z,ms{modelIndex});
                bound = 2.58/sqrt(size(z.OutputData,1));
                acf = r(:,1,1)/max(r(1,1,1),eps);
                xcf = r(:,2,1)/sqrt(max(r(1,1,1)*r(1,2,2),eps));
                rd(modelIndex).model = labels(modelIndex);
                rd(modelIndex).maxAbsAcfExcludingZero = max(abs(acf(2:end)));
                rd(modelIndex).maxAbsInputResidualCorrelation = max(abs(xcf));
                rd(modelIndex).pointwise99Bound = bound;
                rd(modelIndex).whiteAt99 = all(abs(acf(2:end))<bound);
                rd(modelIndex).inputUncorrelatedAt99 = all(abs(xcf)<bound);
            end
            records(row).residualDiagnostics = rd;
            if first && sourceIndex >= 3
                nexttile(tl,(k-1)*2+sourceIndex-2);
                compare(z,ms{:},co);
                title(sprintf('%s source %s, complete block %d (rc %.0f)', ...
                    upper(ax),stamp,D.stats(b).nominalBlock,D.stats(b).amplitude));
                first = false;
            end
        end
    end
end
A.screenedComparisons = records;
flat = struct([]);
for n = 1:numel(records)
    flat(n).sourceStamp = records(n).sourceStamp;
    flat(n).axis = records(n).axis;
    flat(n).nominalBlock = records(n).nominalBlock;
    flat(n).sourceModelRole = records(n).sourceModelRole;
    flat(n).batteryMinimum = records(n).batteryMinimum;
    flat(n).belowStartBatteryThreshold = records(n).belowStartBatteryThreshold;
    flat(n).P1DI_174033 = records(n).fits(1);
    flat(n).P1DI_174633 = records(n).fits(2);
    flat(n).tfest_174033 = records(n).fits(3);
    flat(n).tfest_174633 = records(n).fits(4);
end
writetable(struct2table(flat),fullfile(outputDir,'screened_model_comparison.csv'));
exportgraphics(fig,fullfile(outputDir,'cross_run_comparison.png'),'Resolution',150);
writeJSON(fullfile(outputDir,'real_sysid_audit.json'),A);
save(fullfile(outputDir,'real_sysid_audit.mat'),'A','models','prepared');
for k = 1:numel(A.runs)
    fprintf('%s: %s; %d/%d nominal blocks pass completeness/tracking screen\n', ...
        A.runs(k).stamp,A.runs(k).endReason, ...
        nnz([A.runs(k).blocks.screenedComplete]),numel(A.runs(k).blocks));
end
fprintf('Screened comparison rows: %d. Output: %s\n',numel(records),outputDir);
end

function [blocks,S] = nominalBlocks(T,meta)
ev = string({meta.events.event}); times = [meta.events.t];
beg = find(ev == "phase -> EXCITE");
blocks = cell(1,numel(beg)); S = struct([]);
for n = 1:numel(beg)
    first = beg(n);
    last = find(startsWith(ev,"phase -> ") & times > times(first),1);
    if isempty(last), stop = T.t(end); else, stop = times(last); end
    idx = find(T.t>=times(first) & T.t<stop & T.phase=="EXCITE");
    axisNames = string(meta.opt.axes);
    count = numel(meta.opt.amps)*numel(meta.opt.durs)+1;
    ai = min(numel(axisNames),floor((n-1)/count)+1);
    ax = axisNames(ai); bn = mod(n-1,count)+1;
    if ax == "x", uc = T.rc_a; yc = T.X; else, uc = T.rc_c; yc = T.Y; end
    S(n).axis = ax; S(n).nominalBlock = bn;
    S(n).start = times(first); S(n).stop = stop;
    S(n).duration = stop-times(first);
    S(n).amplitude = max(abs(uc(idx)));
    S(n).exciteRows = numel(idx);
    S(n).validFraction = mean(T.valid(idx)==1);
    S(n).batteryRange = finiteRange(T.bat(idx));
    S(n).belowStartBatteryThreshold = S(n).batteryRange(1)<meta.rig.tello.min_battery;
    S(n).heightRange = finiteRange(T.h(idx));
    S(n).boxHeightRange = finiteRange(T.box_h(idx));
    S(n).truncated = any(contains(ev,"stopped:") & times>=times(first) & times<=stop);
    ue = uc(idx); nz = sign(ue(ue~=0));
    S(n).nonzeroSignSwitches = nnz(diff(nz)~=0);
    sel = find(T.t>=times(first)-0.6 & T.t<=stop+1);
    tt = T.t(sel); y = yc(sel); u = uc(sel); ok = isfinite(y) & T.valid(sel)==1;
    S(n).maxValidMeasurementGap = inf;
    if nnz(ok)>=10
        S(n).maxValidMeasurementGap = max(diff(tt(ok)));
        tg = (tt(1):1/30:tt(end))';
        ug = interp1(tt,u,tg,'previous','extrap');
        yg = interp1(tt(ok),y(ok),tg,'linear','extrap');
        z = iddata(yg-yg(1),ug,1/30,'InputName',"rc_"+ax,'OutputName',upper(ax)+"_cm");
        z.InterSample = 'zoh'; blocks{n} = z;
    end
    bh = S(n).boxHeightRange;
    S(n).screenedComplete = ~S(n).truncated && S(n).validFraction>=0.98 && ...
        S(n).maxValidMeasurementGap<=0.2 && bh(1)>0 && bh(2)/bh(1)<=2.2 && ~isempty(blocks{n});
end
end

function r = finiteRange(v)
v = v(isfinite(v));
if isempty(v), r = [nan nan]; else, r = [min(v) max(v)]; end
end

function writeJSON(path,value)
fid = fopen(path,'w'); assert(fid>=0,'Cannot open output JSON.');
guard = onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
