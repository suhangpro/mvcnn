function parsave(filename, structX)
    vl_xmkdir(fileparts(filename));
    save(filename,'-struct','structX');
end
