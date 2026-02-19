// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Course 1: Lakehouse Fundamentals</li><li class="chapter-item expanded "><a href="course1/overview.html"><strong aria-hidden="true">1.</strong> Overview</a></li><li class="chapter-item expanded "><a href="course1/week1.html"><strong aria-hidden="true">2.</strong> Week 1: Lakehouse Architecture</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course1/lab-lakehouse.html"><strong aria-hidden="true">2.1.</strong> Lab: Lakehouse Concepts</a></li><li class="chapter-item "><a href="course1/lab-workspace.html"><strong aria-hidden="true">2.2.</strong> Lab: Workspace &amp; Catalog</a></li></ol></li><li class="chapter-item expanded "><a href="course1/week2.html"><strong aria-hidden="true">3.</strong> Week 2: Spark Fundamentals</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course1/lab-notebooks.html"><strong aria-hidden="true">3.1.</strong> Lab: Using Notebooks</a></li><li class="chapter-item "><a href="course1/lab-spark.html"><strong aria-hidden="true">3.2.</strong> Lab: Spark Operations</a></li></ol></li><li class="chapter-item expanded "><a href="course1/week3.html"><strong aria-hidden="true">4.</strong> Week 3: Delta Lake &amp; Workflows</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course1/lab-delta.html"><strong aria-hidden="true">4.1.</strong> Lab: Delta Tables</a></li><li class="chapter-item "><a href="course1/lab-workflows.html"><strong aria-hidden="true">4.2.</strong> Lab: Jobs &amp; Workflows</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Course 3: MLOps Engineering</li><li class="chapter-item expanded "><a href="course3/overview.html"><strong aria-hidden="true">5.</strong> Overview</a></li><li class="chapter-item expanded "><a href="course3/week1.html"><strong aria-hidden="true">6.</strong> Week 1: Tracking &amp; Features</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course3/lab-mlflow.html"><strong aria-hidden="true">6.1.</strong> Lab: MLflow Client</a></li><li class="chapter-item "><a href="course3/lab-features.html"><strong aria-hidden="true">6.2.</strong> Lab: Feature Pipeline</a></li></ol></li><li class="chapter-item expanded "><a href="course3/week3.html"><strong aria-hidden="true">7.</strong> Week 2: Training &amp; Serving</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course3/lab-training.html"><strong aria-hidden="true">7.1.</strong> Lab: Model Training</a></li><li class="chapter-item "><a href="course3/lab-serving.html"><strong aria-hidden="true">7.2.</strong> Lab: Inference Server</a></li></ol></li><li class="chapter-item expanded "><a href="course3/week5.html"><strong aria-hidden="true">8.</strong> Week 3: Production &amp; Capstone</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course3/lab-quality.html"><strong aria-hidden="true">8.1.</strong> Lab: Quality Gates</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Course 4: GenAI Engineering</li><li class="chapter-item expanded "><a href="course4/overview.html"><strong aria-hidden="true">9.</strong> Overview</a></li><li class="chapter-item expanded "><a href="course4/week1.html"><strong aria-hidden="true">10.</strong> Week 1: LLM &amp; Prompts</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course4/lab-tokenizer.html"><strong aria-hidden="true">10.1.</strong> Lab: Tokenizer</a></li><li class="chapter-item "><a href="course4/lab-prompts.html"><strong aria-hidden="true">10.2.</strong> Lab: Prompt Templates</a></li></ol></li><li class="chapter-item expanded "><a href="course4/week3.html"><strong aria-hidden="true">11.</strong> Week 2: Vectors &amp; RAG</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course4/lab-embeddings.html"><strong aria-hidden="true">11.1.</strong> Lab: Embeddings</a></li><li class="chapter-item "><a href="course4/lab-rag.html"><strong aria-hidden="true">11.2.</strong> Lab: RAG Pipeline</a></li></ol></li><li class="chapter-item expanded "><a href="course4/week5.html"><strong aria-hidden="true">12.</strong> Week 3: Fine-tune &amp; Production</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="course4/lab-finetuning.html"><strong aria-hidden="true">12.1.</strong> Lab: Fine-Tuning</a></li><li class="chapter-item "><a href="course4/lab-production.html"><strong aria-hidden="true">12.2.</strong> Lab: Production Deployment</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Appendix</li><li class="chapter-item expanded "><a href="sovereign-ai-stack.html"><strong aria-hidden="true">13.</strong> Sovereign AI Stack</a></li><li class="chapter-item expanded "><a href="databricks-setup.html"><strong aria-hidden="true">14.</strong> Databricks Setup</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
