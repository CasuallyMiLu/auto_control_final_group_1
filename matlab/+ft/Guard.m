classdef Guard < handle
    %FT.GUARD  Ctrl-C / crash safety: onCleanup(@() guard.fire()) lands the drone if the program stops while it
    %   is flying (nested-function variables are already gone when onCleanup runs, a handle object is not).
    properties
        tello = []; done = false
    end
    methods
        function fire(o)
            if o.done || isempty(o.tello), return; end
            try
                if o.tello.flying, fprintf('[guard] program stopped while flying: landing\n'); o.tello.land(); end
            catch
            end
            try, o.tello.close(); catch, end
        end
    end
end
