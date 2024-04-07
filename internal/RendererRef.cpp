#define THIS_IS_REF
#include "RendererCPU.h"

namespace Ray {
template class Cpu::Renderer<Ref::SIMDPolicy>;

namespace Ref {
RendererBase *CreateRenderer(const settings_t &s, ILog *log) { return new Cpu::Renderer<Ref::SIMDPolicy>(s, log); }
}
} // namespace Ray
