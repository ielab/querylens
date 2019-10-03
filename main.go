package main

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/hscells/cqr"
	"github.com/hscells/cui2vec"
	"github.com/hscells/groove/analysis"
	"github.com/hscells/groove/combinator"
	"github.com/hscells/groove/eval"
	"github.com/hscells/groove/learning"
	"github.com/hscells/groove/pipeline"
	"github.com/hscells/groove/stats"
	"github.com/hscells/quickumlsrest"
	"github.com/hscells/quickumlsrest/quiche"
	"github.com/hscells/transmute"
	tpipeline "github.com/hscells/transmute/pipeline"
	"github.com/hscells/trecresults"
	"github.com/ielab/searchrefiner"
	"log"
	"net/http"
	"os"
	"sort"
	"time"
)

type QueryLensPlugin struct{}

type lensRequest struct {
	Query    string `json:"query"`
	Language string `json:"language"`
}

type queryVariation struct {
	Query     string  `json:"query"`
	Shape     string  `json:"shape"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	NumRet    float64 `json:"num_ret"`
}

type lensResponse struct {
	Type     string           `json:"type"`
	Queries  []queryVariation `json:"queries"`
	Progress float64          `json:"progress"`
	Message  string           `json:"message"`
}

// Upgrader upgrades a web socket.
var upgrader = websocket.Upgrader{
	ReadBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}
var (
	embeddings cui2vec.Embeddings
	mapping    cui2vec.Mapping
	cache      quickumlsrest.Cache
)

func generateVariations(query cqr.CommonQueryRepresentation, s stats.StatisticsSource, me analysis.MeasurementExecutor, transformations ...learning.Transformation) ([]learning.CandidateQuery, error) {
	return learning.Variations(
		learning.NewCandidateQuery(query, "querylens", nil),
		s,
		me,
		nil,
		transformations...)
}

func wsEvent(ws *websocket.Conn, s searchrefiner.Server, settings searchrefiner.Settings) {

	readWait := 1 * time.Millisecond
	readTicker := time.NewTicker(readWait)

	defer func() {
		readTicker.Stop()
		_ = ws.Close()
	}()

	qrels := trecresults.NewQrelsFile()
	qrels.Qrels["0"] = make(trecresults.Qrels, len(settings.Relevant))
	for _, pmid := range settings.Relevant {
		qrels.Qrels["0"][pmid.String()] = &trecresults.Qrel{Topic: "0", Iteration: "0", DocId: pmid.String(), Score: 1}
	}
	eval.RelevanceGrade = 0

	for {
		select {
		case <-readTicker.C:
			var request lensRequest
			err := ws.ReadJSON(&request)
			if err != nil {
				log.Println("read:", err)
				return
			}
			rawQuery := request.Query
			lang := request.Language

			p := make(map[string]tpipeline.TransmutePipeline)
			p["medline"] = transmute.Medline2Cqr
			p["pubmed"] = transmute.Pubmed2Cqr

			compiler := p["medline"]
			if v, ok := p[lang]; ok {
				compiler = v
			} else {
				lang = "medline"
			}

			log.Printf("recieved a query %s in format %s\n", rawQuery, lang)

			cq, err := compiler.Execute(rawQuery)
			if err != nil {
				log.Println("read:", err)
				return
			}
			repr, err := cq.Representation()
			if err != nil {
				log.Println("read:", err)
				return
			}

			err = ws.WriteJSON(lensResponse{
				Type:    "message",
				Message: "Generating variations.",
			})
			if err != nil {
				log.Println("write:", err)
				return
			}

			candidate := learning.NewCandidateQuery(repr.(cqr.CommonQueryRepresentation), "0", nil)

			variations, err := generateVariations(repr.(cqr.CommonQueryRepresentation), s.Entrez, analysis.NewMemoryMeasurementExecutor(),
				learning.NewMeSHExplosionTransformer(),
				learning.NewLogicalOperatorTransformer(),
				learning.NewFieldRestrictionsTransformer(),
				learning.NewMeshParentTransformer(),
				learning.NewClauseRemovalTransformer(),
				learning.Newcui2vecExpansionTransformer(embeddings, mapping, cache),
			)

			err = ws.WriteJSON(lensResponse{
				Type:    "message",
				Message: "Predicting most effective variation.",
			})
			if err != nil {
				log.Println("write:", err)
				return
			}

			// selector is a quickrank candidate selector configured to only select to a depth of one.
			ltrModel := learning.NewQuickRankQueryCandidateSelector(
				searchrefiner.ServerConfiguration.Config.Options["QuickRank"].(string),
				map[string]interface{}{
					"model-in":    "plugin/querylens/balanced.xml",
					"test-metric": "DCG",
					"test-cutoff": 1,
					"scores":      uuid.New().String(),
				},
				learning.QuickRankCandidateSelectorMaxDepth(1),
				learning.QuickRankCandidateSelectorStatisticsSource(s.Entrez),
			)
			ltrCandidate, _, err := ltrModel.Select(candidate, variations)
			if err != nil {
				log.Println("read:", err)
				return
			}

			err = ws.WriteJSON(lensResponse{
				Type:    "message",
				Message: "Evaluating queries.",
			})
			if err != nil {
				log.Println("write:", err)
				return
			}

			var backend func(cqr.CommonQueryRepresentation) (string, error)

			queries := make([]queryVariation, len(variations))
			switch lang {
			case "pubmed":
				backend = transmute.CompileCqr2PubMed
			default:
				backend = transmute.CompileCqr2Medline
			}
			qc := combinator.NewFileQueryCache("file_cache")

			for i := range variations {
				q, err := backend(variations[i].Query)
				if err != nil {
					log.Println("write:", err)
					return
				}

				pq := pipeline.NewQuery("0", "0", variations[i].Query)
				t, _, err := combinator.NewLogicalTree(pq, s.Entrez, qc)
				if err != nil {
					log.Println("write:", err)
					return
				}
				results := t.Documents(qc).Results(pq, "0")

				queries[i] = queryVariation{
					Query:     q,
					Shape:     "circle",
					Precision: eval.Precision.Score(&results, qrels.Qrels["0"]),
					Recall:    eval.Recall.Score(&results, qrels.Qrels["0"]),
					F1:        eval.F1Measure.Score(&results, qrels.Qrels["0"]),
					NumRet:    float64(len(results)),
				}

				fmt.Println(queries[i].Precision, queries[i].Recall, queries[i].F1, eval.NumRel.Score(&results, qrels.Qrels["0"]))

				err = ws.WriteJSON(lensResponse{
					Type:     "executing",
					Progress: (float64(i) / float64(len(variations))) * 100,
				})
				if err != nil {
					log.Println("write:", err)
					return
				}
			}

			err = ws.WriteJSON(lensResponse{
				Type:    "message",
				Message: "Evaluating predictions.",
			})
			if err != nil {
				log.Println("write:", err)
				return
			}

			ltrQuery, err := backend(candidate.Query)
			if err != nil {
				log.Println("write:", err)
				return
			}
			ltrPq := pipeline.NewQuery("0", "0", ltrCandidate.Query)
			t1, _, err := combinator.NewLogicalTree(ltrPq, s.Entrez, qc)
			if err != nil {
				log.Println("write:", err)
				return
			}
			ltrResuts := t1.Documents(qc).Results(ltrPq, "0")
			queries = append(queries, queryVariation{
				Query:     ltrQuery,
				Shape:     "triangle",
				Precision: eval.Precision.Score(&ltrResuts, qrels.Qrels["0"]),
				Recall:    eval.Recall.Score(&ltrResuts, qrels.Qrels["0"]),
				F1:        eval.F1Measure.Score(&ltrResuts, qrels.Qrels["0"]),
				NumRet:    float64(len(ltrResuts)),
			})

			pq := pipeline.NewQuery("0", "0", candidate.Query)
			t2, _, err := combinator.NewLogicalTree(pq, s.Entrez, qc)
			if err != nil {
				log.Println("write:", err)
				return
			}
			origResuts := t2.Documents(qc).Results(ltrPq, "0")
			queries = append(queries, queryVariation{
				Query:     rawQuery,
				Shape:     "cross",
				Precision: eval.Precision.Score(&origResuts, qrels.Qrels["0"]),
				Recall:    eval.Recall.Score(&origResuts, qrels.Qrels["0"]),
				F1:        eval.F1Measure.Score(&origResuts, qrels.Qrels["0"]),
				NumRet:    float64(len(origResuts)),
			})

			sort.Slice(queries, func(i, j int) bool {
				return queries[i].F1 < queries[j].F1
			})

			err = ws.WriteJSON(lensResponse{
				Type:    "queries",
				Queries: queries,
			})
			if err != nil {
				log.Println("write:", err)
				return
			}
		}
	}
}

func handleVariations(s searchrefiner.Server, c *gin.Context) {
	ws, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Println(err)
		c.Status(http.StatusInternalServerError)
		return
	}
	go wsEvent(ws, s, searchrefiner.GetSettings(s, c))
}

func (QueryLensPlugin) Serve(s searchrefiner.Server, c *gin.Context) {
	if embeddings == nil {
		log.Println("loading cui2vec components")
		f, err := os.OpenFile(searchrefiner.ServerConfiguration.Config.Options["Cui2VecEmbeddings"].(string), os.O_RDONLY, 0644)
		if err != nil {
			c.Status(http.StatusInternalServerError)
			return
		}
		defer f.Close()

		log.Println("loading mappings")
		mapping, err = cui2vec.LoadCUIMapping(searchrefiner.ServerConfiguration.Config.Options["Cui2VecMappings"].(string))
		if err != nil {
			c.Status(http.StatusInternalServerError)
			return
		}

		log.Println("loading model")
		embeddings, err = cui2vec.NewPrecomputedEmbeddings(f)
		if err != nil {
			c.Status(http.StatusInternalServerError)
			return
		}

		log.Println("loading quiche cache")
		cache, err = quiche.Load(searchrefiner.ServerConfiguration.Config.Options["Quiche"].(string))
		if err != nil {
			c.Status(http.StatusInternalServerError)
			return
		}
	}

	if c.Query("lens") == "y" {
		handleVariations(s, c)
		return
	}
	c.Render(http.StatusOK, searchrefiner.RenderPlugin(searchrefiner.TemplatePlugin("plugin/querylens/index.html"), nil))
}

func (QueryLensPlugin) PermissionType() searchrefiner.PluginPermission {
	return searchrefiner.PluginUser
}

func (QueryLensPlugin) Details() searchrefiner.PluginDetails {
	return searchrefiner.PluginDetails{
		Title:       "QueryLens",
		Description: "(Semi)-Automatic query refinement and exploration.",
		Author:      "Harry Scells",
		Version:     "02.Oct.2019",
		ProjectURL:  "https://github.com/hscells/querylens",
	}
}

var Querylens QueryLensPlugin
