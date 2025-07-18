import os

from pybind11.setup_helpers import STD_TMPL, WIN, Pybind11Extension
from setuptools import setup

for float_type in ["float", "double"]:
    tmpl_args = dict(float_type=float_type)

    import mako.lookup

    tmpl_fp = "cutde/cpp_backend.cpp"
    tmpl_name = os.path.basename(tmpl_fp)
    lookup = mako.lookup.TemplateLookup(directories=["cutde"])
    tmpl = lookup.get_template(tmpl_name)
    try:
        rendered_tmpl = tmpl.render(**tmpl_args, backend="cpp", preamble="")
        assert isinstance(rendered_tmpl, str)
    except:  # noqa: E722
        # bare except is okay because we re-raise immediately
        import mako.exceptions

        print(mako.exceptions.text_error_template().render())
        raise

    rendered_fp = os.path.join(
        os.path.dirname(tmpl_fp), f".rendered.{float_type}.{tmpl_name}"
    )
    with open(rendered_fp, "w") as f:
        f.write(rendered_tmpl)

OPENMP_FLAG = "/openmp" if WIN else "-fopenmp"

ext_modules = [
    Pybind11Extension(
        f"cutde.cpp_backend_{float_type}",
        [f"cutde/.rendered.{float_type}.cpp_backend.cpp"],
        extra_compile_args=[OPENMP_FLAG, STD_TMPL.format("17")],
        extra_link_args=[] if WIN else [OPENMP_FLAG],
    )
    for float_type in ["float", "double"]
]

# We need to call setup() for compilation.
# The rest of the metadata is taken automatically from pyproject.toml.
setup(ext_modules=ext_modules)
